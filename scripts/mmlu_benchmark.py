import dataclasses
import logging
import os
import pprint
import shutil
from contextlib import ExitStack
from pathlib import Path

import fire
import torch.cuda
import torch.distributed as dist
from torch.optim import AdamW, lr_scheduler

# from torch.profiler import ProfilerActivity, profile

from finetune.args import TrainArgs
from finetune.checkpointing import Checkpointer
from finetune.data.data_loader import build_data_loader
from finetune.data.interleaver import InterleavedTokenizer, Interleaver
from finetune.distributed import (
    BACKEND,
    avg_aggregate,
    get_rank,
    get_world_size,
    is_torchrun,
    set_device,
)
from finetune.eval import evaluate
from finetune.loss import compute_loss_with_mask
from finetune.mixed_precision import (
    downcast_mixed_precision,
    prepare_mixed_precision,
    upcast_mixed_precision,
)
from finetune.monitoring.metrics_logger import (
    MetricsLogger,
    eval_log_msg,
    get_eval_logs,
    get_train_logs,
    train_log_msg,
)
from finetune.monitoring.utils import set_logger
from finetune.utils import TrainState, logged_closing, set_random_seed
from finetune.wrapped_model import get_fsdp_model
from moshi.models import loaders
from moshi.utils.sampling import sample_token
import sphn
import numpy as np
from glob import glob
import argparse
import string
import json
import random
import re

logger = logging.getLogger("train")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)


def train(config: str):
    args: TrainArgs = TrainArgs.load(config, drop_extra_fields=False)
    set_logger(logging.INFO)

    with ExitStack() as exit_stack:
        _train(args, exit_stack)
    logger.info("Closed everything!")

LETTER_INDICES = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda')
    parser.add_argument("-f", "--file", type=str, required=True, help="Path to MMLU jsonl file or directory containing jsonl or pt files")
    parser.add_argument("--num-few-shot", type=int, default=5, help="Number of few-shot examples")
    parser.add_argument("--num-options", type=int, default=4, help="Number of options to evaluate (4 for MMLU, 10 for MMLU-Pro)")
    parser.add_argument("--model-type", type=str, default="new", help="Model type to use - hf, moshi, new")
    return parser.parse_args()

def format_example(example, topic, include_answer=True, is_few_shot=False):
    """Format a single example following HuggingFace format"""
    # Get all available options
    choices = [example[f'answer_{chr(ord("a") + i)}'] for i in range(len(LETTER_INDICES)) 
              if f'answer_{chr(ord("a") + i)}' in example]
    
    query = example['question'] + "\n"
    query += "".join([f"Option {key}. {choice}\n" for key, choice in zip(LETTER_INDICES[:len(choices)], choices)])
    
    if include_answer:
        query += f"The correct answer is {example['correct_answer']} \n\n"
    else:
        query += "The correct answer is"

    return query

def create_few_shot_prompt(examples, question, topic):
    """Create a prompt with few-shot examples and the target question"""
    prompt = f"The following are multiple choice questions (with answers) about {topic}.\n\n"
    # Add few-shot examples
    for example in examples:
        prompt += format_example(example, topic, include_answer=True, is_few_shot=True)
    # Add the actual question
    prompt += format_example(question, topic, include_answer=False, is_few_shot=False)
    prompt = prompt.lower().translate(str.maketrans('', '', string.punctuation))
    prompt = re.sub(r"\s+", " ", prompt)
    prompt = prompt.strip()

    return prompt

def merge_tensors(tensors):
    for i in range(len(tensors)-1):
        for j in (0,1,9):
            tensors[i][j, -1] = tensors[i+1][j, 1]

    for i in range(1, len(tensors)):
        tensors[i] = tensors[i][:, 2:]

    return torch.cat(tensors, dim=-1)

def evaluate_subject(model, tokenizer, data, args):
    """Evaluate model on a single subject's data"""
    few_shot_pool = data.copy()
    random.shuffle(few_shot_pool)

    # Extract topic from the first example
    topic = data[0].get('topic', 'general knowledge').replace('_', ' ')

    cnt_correct = 0
    cnt_total = 0

    # Get token IDs for answer choices (with space prefix for few-shot examples)
    answer_tokens = {
        letter.lower(): tokenizer.encode(f' {letter.lower()}', add_special_tokens=False)[0] if args.model_type != "moshi" else tokenizer.encode(f' {letter.lower()}')[1]
        for letter in LETTER_INDICES[:args.num_options]
    }

    # Evaluate on first 150 examples
    for j in range(min(150, len(data))):
        item = data[j]
        
        # Select few-shot examples (excluding current question)
        prompt = ""
        if args.model_type == "hf":
            few_shot_examples = []
            for example in few_shot_pool:
                if example != item and len(few_shot_examples) < args.num_few_shot:
                    few_shot_examples.append(example)

            # Create prompt with few-shot examples
            prompt = create_few_shot_prompt(few_shot_examples, item, topic)
        else:
            idxs = [0]
            population = [x for x in range(len(data)) if x not in (0, j)]
            idxs += random.sample(population, args.num_few_shot)
            idxs += [j]
            to_merge = [data[i]['tensor'].clone() for i in idxs]
            inputs = merge_tensors(to_merge)

        # Tokenize and prepare input
        if args.model_type == "hf":
            inputs = tokenizer(prompt, return_tensors="pt").to(args.device)
        else:
            stop_set = set(answer_tokens.values())
            row0 = inputs[0, :]
            cut_index = inputs.shape[-1]
            for i in range(row0.size(0)-1, -1, -1):
                if row0[i].item() in stop_set:
                    cut_index = i
                    item['correct_answer'] = next((k for k, v in answer_tokens.items() if v == row0[i].item()), None)
                    break
                
            inputs = inputs[:, 1:cut_index+1] 
            inputs = inputs.unsqueeze(0).to(args.device).long()
            if inputs.shape[-1] > 3000:
                print(f"Skipping example {j} due to length")
                continue
            if item.get('correct_answer') is None:
                print(f"Skipping example {j} due to missing correct answer")
                continue
        
        # Get model's predictions
        with torch.no_grad():
            if args.model_type == "hf":
                outputs = model(**inputs)
                logits = outputs.logits[:, -1, :]  # Get logits for the last token
            else:
                output = model(codes=inputs)
                text_logits - output.text_logits
                if text_logits.dim() == 4:
                    text_logits = text_logits.squeeze(1)
                logits = text_logits[:, -1, :]  # Get logits for the last token

        # Get the three most likely predicted tokens
        top_k = logits.topk(3, dim=-1)
        for i in range(3):
            token_id = top_k.indices[0, i].item()
            token = tokenizer.decode([token_id])
            score = top_k.values[0, i].item()
            print(f"Top {i+1} predicted token: {token}, Score: {score}")

        # Get scores for all options
        scores = {letter: logits[0, token_id].item() for letter, token_id in answer_tokens.items()}
        pred_answer = max(scores.items(), key=lambda x: x[1])[0]
        for letter, score in scores.items():
            print(f"Score for {letter}: {score}")
        print(f"Correct answer: {item['correct_answer']}, Predicted answer: {pred_answer}")
        if pred_answer.lower() == item["correct_answer"].lower():
            cnt_correct += 1
        cnt_total += 1

        # print out the text input
        # print(inputs)
        # text = tokenizer.decode(inputs[0, 0, :].tolist())
        # text = text.replace("<pad>", "  ").replace("<unk>", "  ")
        # text = text.replace("<|im_end|>", "").replace("<|endoftext|>", "    ")
        # print(f"Text input: {text}")

    return {
        'topic': topic,
        'accuracy': cnt_correct/cnt_total,
        'correct': cnt_correct,
        'total': cnt_total
    }


def _train(args: TrainArgs, exit_stack: ExitStack):
    # 1. Initial setup and checks
    set_random_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    # Init NCCL
    if "LOCAL_RANK" in os.environ:
        set_device()
        logger.info("Going to init comms...")

        dist.init_process_group(backend=BACKEND)
    else:
        logger.error(
            "PyTorch environment is not correctly initialized. This message should only be displayed when testing."
        )

    # 4.1 Load function calling audio encoder and tokenizer
    main_logger_info("Loading Mimi and Moshi...")
    checkpoint_info = loaders.CheckpointInfo.from_hf_repo(
        hf_repo=args.moshi_paths.hf_repo_id,
        moshi_weights=args.moshi_paths.moshi_path,
        mimi_weights=args.moshi_paths.mimi_path,
        tokenizer=args.moshi_paths.tokenizer_path,
        config_path=args.moshi_paths.config_path,
    )

    lm_config = (
        loaders._lm_kwargs
        if checkpoint_info.raw_config is None
        else checkpoint_info.raw_config
    )
    lm_config["lora"] = args.lora.enable
    lm_config["lora_rank"] = args.lora.rank
    lm_config["lora_scaling"] = args.lora.scaling

    mimi = checkpoint_info.get_mimi(device="cuda")
    mimi.eval()
    for p in mimi.parameters():
        p.requires_grad = False

    # 4.2 Load and shard model, prepare interleaver for audio/text tokens.
    model = get_fsdp_model(args, checkpoint_info)

    spm = checkpoint_info.get_text_tokenizer()

    # 6. Load model
    # Define mixed precision
    param_dtype = getattr(torch, args.param_dtype)
    optim_dtype = torch.float32

    assert args.lora is not None, "`args.lora` should be set to a valid value."
    # 9. Prepare mixed precision
    prepare_mixed_precision(
        model.parameters(), param_dtype=param_dtype, optim_dtype=optim_dtype
    )

    # 11. train!
    tokenizer = checkpoint_info.get_text_tokenizer()

    model.eval()

    # Get list of files to process
    if os.path.isdir(args.file):
        if args.model_type == "hf":
            files = glob(os.path.join(args.file, "*.jsonl"))
        else:
            files = glob(os.path.join(args.file, "*.pt"))
            files = [f for f in files if 'audio' not in os.path.basename(f) and 'text' not in os.path.basename(f)]
        print(f"Found {len(files)}  files in directory")
    else:
        files = [args.file]
        print("Processing single file")

    all_results = []
    
    # Process each file
    for file_path in files:
        print(f"\nProcessing {os.path.basename(file_path)}...")
        if args.model_type == "hf":
            with open(file_path, "r", encoding="utf-8") as file:
                data = [json.loads(line) for line in file]
        else:
            data = torch.load(file_path, weights_only=False)

        result = evaluate_subject(model, tokenizer, data, args)
        all_results.append(result)
        print(f"Accuracy for {result['topic']}: {result['accuracy']:.3f} ({result['correct']}/{result['total']})")
        break

    # Print overall results
    if len(all_results) > 1:
        print("\nOverall Results:")
        print("=" * 50)
        total_correct = sum(r['correct'] for r in all_results)
        total_questions = sum(r['total'] for r in all_results)
        print(f"Average Accuracy: {total_correct/total_questions:.3f}")
        print("\nResults by subject:")
        for result in sorted(all_results, key=lambda x: x['accuracy'], reverse=True):
            print(f"{result['topic']}: {result['accuracy']:.3f}")

if __name__ == "__main__":
    """See README.md for usage."""
    fire.Fire(train)
