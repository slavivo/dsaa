import logging
from typing import Iterator, Dict

import torch
import torch.cuda
import torch.distributed as dist
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel

from finetune.args import TrainArgs

from .data.data_loader import Batch
from .distributed import get_rank, get_world_size
from .loss import compute_loss_with_mask
from .utils import TrainState

import torch.nn.functional as F

logger = logging.getLogger("eval")


def main_logger_info(message: str) -> None:
    if get_rank() == 0:
        logger.info(message)

def eval_loss_evaluate(
        model: FullyShardedDataParallel,
        data_loader: Iterator[Batch],
        args: TrainArgs,
        text_tokenizer,
        num_batches: int = 50
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_samples = torch.tensor([0], device="cuda", dtype=torch.long)

    text_loss = torch.tensor(0.0).cuda()
    audio_loss = torch.tensor(0.0).cuda()
    for batch in data_loader:
        num_samples += 1
        if num_samples % 100 == 0:
            print(f"{num_samples.item()} / {num_batches}")
        if num_samples > num_batches // get_world_size():
            break
        with torch.no_grad():
            codes = batch.codes
            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            output = model(codes=codes, condition_tensors=condition_tensors)
            text_loss += compute_loss_with_mask(
                output.text_logits,
                codes[:, : model.audio_offset],
                output.text_mask,
                mode="text",
                text_padding_weight=0.1,
                text_padding_ids={
                    model.text_padding_token_id,
                    model.end_of_text_padding_id,
                },
            )
            audio_loss += compute_loss_with_mask(
                output.logits,
                codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                output.mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier,
            )
    eval_loss = text_loss + audio_loss

    return num_samples, eval_loss, text_loss, audio_loss

@torch.no_grad()
def compute_loglikelihood_from_codes(model, context_codes, continuation_codes, text_tokenizer, pad_token_id=220, pad_weight=.5, audio_to_text_weight=0.1):
    """
    Stable version: same logic as the simple one-stream MMLU scorer,
    extended to text + multi-audio streams.
    """
    # Combine context + continuation along time axis
    full_codes = torch.cat([context_codes, continuation_codes], dim=-1)  # [1, 17, total_T]
    # print(f"Full input: {text_tokenizer.decode(full_codes[0,0,:].tolist())}")

    # Forward pass
    output = model(codes=full_codes, condition_tensors=None)
    text_logits  = output.text_logits[0, 0]        # [T, text_card]
    audio_logits = output.logits[0]                # [K, T, card]

    # Log-softmax
    text_logprobs  = F.log_softmax(text_logits,  dim=-1)
    audio_logprobs = F.log_softmax(audio_logits, dim=-1)

    prefix_len = context_codes.size(-1)
    cont_len   = continuation_codes.size(-1)

    start = prefix_len - 1
    end   = start + cont_len

     # --- Text continuation ---
    cont_text_logprobs = text_logprobs[start:end-1]       # predict next tokens
    cont_text_ids      = continuation_codes[0, 0, :cont_len-1]  # [T]

    # Gather the logprobs corresponding to target tokens
    token_logprobs = cont_text_logprobs.gather(1, cont_text_ids.unsqueeze(1)).squeeze(1)
    text_ll = token_logprobs.sum().item()

    # --- Audio continuation (average over streams) ---
    cont_audio_logprobs = audio_logprobs[:, start:end-1, :]   # [K, cont_len-1, card]
    cont_audio_ids = continuation_codes[
        0, model.audio_offset : model.audio_offset + model.dep_q, :cont_len-1
    ]                                                         # [K, cont_len-1]

    # Loop per stream for clarity and safety
    audio_scores = []
    for k in range(model.dep_q):
        stream_logits = cont_audio_logprobs[k]                # [cont_len-1, card]
        stream_ids    = cont_audio_ids[k]
        valid = (stream_ids >= 0) & (stream_ids < stream_logits.size(1))
        if valid.any():
            ll = stream_logits[valid].gather(
                1, stream_ids[valid].unsqueeze(1)
            ).sum().item()
            audio_scores.append(ll)
    audio_ll = sum(audio_scores) / max(1, len(audio_scores))

    # Combine weighted
    print(f"Text ll: {text_ll}, audio ll: {audio_ll}")
    total_ll = (1 - audio_to_text_weight) * text_ll + audio_to_text_weight * audio_ll
    return total_ll

import torch

def mmlu_evaluate(
        model: FullyShardedDataParallel,
        text_tokenizer,
        data_loader: Iterator[Batch],
        args: TrainArgs,
        num_batches: int = 50,
        audio_to_text_weight: float = 0.1
    ):
    print("=== Loading prefix and choice batches ===")
    choice_codes = []
    choice_avg = {}

    # Preload the first 4 + 2 + 5 * 57 batches: choices A-D, one and few-shot prepends, and 5-shot samples
    preload_batches = []
    for i in range(4 + 2 + 5 * 57):
        batch = next(data_loader)
        preload_batches.append(batch)

    for i in range(4):
        choice_avg[i] = []
        codes = preload_batches[i].codes[:,:,:33]
        choice_codes.append(codes)
        print(f"codes: {len(choice_codes[-1][0,0,:].tolist())}")
        print(f"choice code: {text_tokenizer.decode(choice_codes[-1][0,0,:].tolist())}")
        # print(f"choice code: {choice_codes[-1][0,0,:].tolist()}")

    codes = preload_batches[4].codes
    prepend_codes = codes
    print(f"few shot codes: {text_tokenizer.decode(prepend_codes[0,0,:].tolist())}")
    
    few_shot_codes = {}
    for i in range(6, 4 + 2 + 5 * 57):
        batch = preload_batches[i]
        golden = int(batch.extra[0]['number'])
        subject = int(batch.extra[0]['subject_index'])
        codes = batch.codes
        codes = torch.cat([codes, choice_codes[golden]], dim=-1)  # append correct choice to the end
        if subject not in few_shot_codes:
            few_shot_codes[subject] = codes
        else:
            few_shot_codes[subject] = torch.cat([few_shot_codes[subject], codes], dim=-1)

        # print(f"5-shot codes {i-6}: {text_tokenizer.decode(codes[0,0,:].tolist())}")

    print("Loaded A–D choices and one-shot prepend.\n")

    # Initialize metrics
    num_correct = torch.tensor([0], device="cuda", dtype=torch.long)
    num_seen = torch.tensor([0], device="cuda", dtype=torch.long)
    choices_count = [0, 0, 0, 0]
    choices = ["A", "B", "C", "D"]

    print("=== Starting evaluation ===")

    # Iterate over remaining batches
    for batch_idx, batch in enumerate(data_loader, start=6):
        if batch_idx >= num_batches:
            break

        codes = batch.codes  # (B, 17, T)
        golden = int(batch.extra[0]['number'])    # correct answer index (0–3)
        subject = int(batch.extra[0]['subject_index'])  # subject index to get few-shot codes

        # print(f"Codes first stream {len(codes[0,0,:].tolist())}: {codes[0,0,:].tolist()}")
        seq = codes[0,0]
        idx = (seq == -1).nonzero(as_tuple=True)[0]
        if len(idx) > 0:
            codes = codes[:, :, :idx[0]]

        # Combine prefix + few-shot codes + question codes
        context_codes = torch.cat([prepend_codes, few_shot_codes[subject], codes], dim=-1)

        # Compute score for each continuation choice
        scores = []
        for i, cont_codes in enumerate(choice_codes):
            score = compute_loglikelihood_from_codes(
                model,
                context_codes=context_codes,
                continuation_codes=cont_codes,
                text_tokenizer=text_tokenizer,
                pad_token_id=model.text_padding_token_id,
                pad_weight=args.text_padding_weight,
                audio_to_text_weight=audio_to_text_weight
            )
            # if i == 0:
            #     score *= 1.035  # calibration factor for choice A
            print(f"Score: {score}")
            choice_avg[i].append(score)
            scores.append(score)

        pred = int(torch.tensor(scores).argmax())
        choices_count[pred] += 1
        num_correct += (pred == golden)
        num_seen += 1

        print(f"[Batch {batch_idx}] Predicted {choices[pred]}, correct {choices[golden]}")

    print(f"Avg scores: A: {sum(choice_avg[0])/len(choice_avg[0])}, B: {sum(choice_avg[1])/len(choice_avg[1])}, C: {sum(choice_avg[2])/len(choice_avg[2])}, D: {sum(choice_avg[3])/len(choice_avg[3])}")
    print(f"Choices distribution: A: {choices_count[0]}, B: {choices_count[1]}, C: {choices_count[2]}, D: {choices_count[3]}")
    return num_seen, num_correct

def slm_evaluate(
        model: FullyShardedDataParallel,
        data_loader: Iterator[Batch],
        args: TrainArgs,
        text_tokenizer,
        num_batches: int = 50
):
    num_samples = torch.tensor([0], device="cuda", dtype=torch.long)
    num_correct = torch.tensor([0], device="cuda", dtype=torch.long)

    prev_loss = None
    total_text_loss = 0.0
    total_audio_loss = 0.0
    for batch in data_loader:
        if num_samples > num_batches // get_world_size():
            break
        with torch.no_grad():
            codes = batch.codes
            # Mask the text stream input
            codes[0, 0, :] = -1
            # print(f"Full input: {text_tokenizer.decode(codes[0,0,:].tolist())}")
            correct = batch.extra
            condition_tensors = None
            if batch.condition_attributes is not None:
                condition_tensors = model.condition_provider.prepare(
                    batch.condition_attributes
                )

            output = model(codes=codes, condition_tensors=condition_tensors)
            """
            text_loss = compute_loss_with_mask(
                output.text_logits,
                codes[:, : model.audio_offset],
                output.text_mask,
                mode="text",
                text_padding_weight=args.text_padding_weight,
                text_padding_ids={
                    model.text_padding_token_id,
                    model.end_of_text_padding_id,
                }
            )
            """
            audio_loss = compute_loss_with_mask(
                output.logits,
                codes[:, model.audio_offset : model.audio_offset + model.dep_q],
                output.mask,
                mode="audio",
                first_codebook_weight_multiplier=args.first_codebook_weight_multiplier
            )
            # total_text_loss += text_loss.item()
            # total_audio_loss += audio_loss.item()
            # loss = text_loss + audio_loss
            loss = audio_loss

            if prev_loss is None:
                prev_loss = loss.item()
                continue
            
            num_samples += 1

            if num_samples % 100 == 0:
                print(f"{num_samples.item()} / {num_batches}")
                
            if loss.item() < prev_loss:
                if int(correct[0]['number']):
                    num_correct += 1
            else:
                if not int(correct[0]['number']):
                    num_correct += 1
            prev_loss = None
    # print(f"Avg text loss: {total_text_loss / (num_samples.item() * 2)}, avg audio loss: {total_audio_loss / (num_samples.item() * 2)}")

    return num_samples, num_correct

def evaluate(
        model: FullyShardedDataParallel,
        text_tokenizer,
        eval_data_loaders: Dict[str, Iterator[Batch]],
        state: TrainState,
        args: TrainArgs
):
    model.eval()

    if args.do_mmlu:
        num_samples, num_correct = mmlu_evaluate(
            model, 
            text_tokenizer, 
            eval_data_loaders['mmlu'], 
            args,
            args.eval_batches
        )
        dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
        mmlu_acc = num_correct / num_samples
        print(f"MMLU acc: {mmlu_acc.item()}, n samples: {num_samples.item()}")
        state.this_mmlu_acc = mmlu_acc.item()

    if args.do_slm:
        for mode in ['swuggy', 'sblimp', 'ssc']:
            num_samples, num_correct = slm_evaluate(model, eval_data_loaders[mode], args, text_tokenizer, args.eval_batches)
            dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
            slm_acc = num_correct / num_samples
            print(f"Mode: {mode}, acc: {slm_acc.item()}, n samples: {num_samples.item()}")
            if mode == 'swuggy':
                state.this_swuggy_acc = slm_acc.item()
            elif mode == 'sblimp':
                state.this_sblimp_acc = slm_acc.item()
            else:
                state.this_ssc_acc = slm_acc.item()

    if args.do_cd:
        for cd_dataset in args.data.cd_data:
            num_samples, num_correct = slm_evaluate(model, eval_data_loaders[cd_dataset], args, text_tokenizer, args.eval_batches)
            dist.all_reduce(num_samples, op=dist.ReduceOp.SUM)
            dist.all_reduce(num_correct, op=dist.ReduceOp.SUM)
            cd_acc = num_correct / num_samples
            print(f"CD Dataset: {cd_dataset}, acc: {cd_acc.item()}, n samples: {num_samples.item()}")
            if state.this_cd_acc is None:
                state.this_cd_acc = {}
            state.this_cd_acc[cd_dataset] = cd_acc.item()

    if args.do_eval_loss:
        num_samples, eval_loss, text_loss, audio_loss = eval_loss_evaluate(
            model, eval_data_loaders['eval_loss'], args, text_tokenizer, args.eval_batches
        )

        all_num_samples = [torch.zeros_like(num_samples) for _ in range(get_world_size())]

        torch.distributed.all_gather(all_num_samples, num_samples)

        total_num_samples = int(torch.tensor(all_num_samples).sum().item())
        # sum loss
        main_logger_info("Eval finished!")

        dist.all_reduce(eval_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(text_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(audio_loss, op=dist.ReduceOp.SUM)
        text_loss /= total_num_samples
        audio_loss /= total_num_samples
        eval_loss /= total_num_samples

        state.this_eval_loss = eval_loss.item()
        state.this_eval_perplexity = (2**eval_loss).item()
        state.this_audio_loss = audio_loss.item()
        state.this_text_loss = text_loss.item()

    # train mode!
    model.train()
