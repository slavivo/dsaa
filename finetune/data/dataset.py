import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal

import numpy as np
import sphn
import torch.distributed as dist

from finetune.distributed import get_rank

from .interleaver import InterleavedTokenizer, Sample
from .args import DataArgs
from .augmenter import WaveAugmenter, CodeAugmenter

import json
import random
import tempfile
import os
import atexit

logger = logging.getLogger("dataset")


AudioChunkPath = tuple[str, float]
_LOADED_DATASETS: dict[Path, list[AudioChunkPath]] = {}


def main_logger_info(message: str) -> None:
    if dist.is_initialized() and get_rank() == 0:
        logger.info(message)


def load_file(path: Path, world_size: int, rank: int) -> list[str]:
    lines = []
    with path.open() as f:
        for idx, line in enumerate(f):
            if not idx % world_size == rank:
                continue
            lines.append(line)
    return lines


def maybe_load_local_dataset(
    path: Path, rank: int, world_size: int, instruct_tokenizer: InterleavedTokenizer
) -> list[AudioChunkPath]:
    if path in _LOADED_DATASETS:
        return _LOADED_DATASETS[path]

    duration = instruct_tokenizer.duration_sec
    main_logger_info(f"Loading {path} ...")
    lines: list[str] = load_file(path, rank=rank, world_size=world_size)

    chunks: list[AudioChunkPath] = []
    for line in lines:
        data = json.loads(line)
        start_sec = 0
        while start_sec < data["duration"]:
            chunks.append((data["path"], start_sec))
            start_sec += duration

    main_logger_info(f"{path} loaded and chunked.")
    _LOADED_DATASETS[path] = chunks

    return _LOADED_DATASETS[path]


@dataclass
class DataDir:
    path: Path

    @property
    def jsonl_files(self):
        assert self.path.exists(), f"Make sure that {self.path} exists"
        jsonl_files = list(self.path.rglob("*jsonl"))
        assert len(jsonl_files) > 0, (
            f"{self.path} does not seem to have any files ending with '.jsonl'"
        )
        return jsonl_files


@dataclass
class DataFile:
    path: Path

    @property
    def jsonl_files(self):
        assert self.path.exists(), f"Make sure that {self.path} exists"
        return [self.path]


def parse_data_sources(
    pretrain_data: str,
) -> tuple[list[DataDir | DataFile], list[float]]:
    seen: set[str] = set()
    sources: list[DataDir | DataFile] = []
    weights: list[float] = []

    sample_sources = pretrain_data

    for source in sample_sources.strip().split(","):
        if not source:
            continue

        source_items = source.strip().split(":")
        if len(source_items) == 1:
            path_ = source_items[0]
            weight = 1.0
        elif len(source_items) == 2:
            path_, weight_ = source_items
            weight = float(weight_)
        else:
            raise ValueError(
                f"{source} is not correctly formatted. Make sure to format each data source "
                "as <path/to/data>:<weight> or just <path/to/data>"
            )

        assert path_ not in seen, (
            f"{path_} seems to be duplicated. Make sure to only add it once."
        )
        assert weight > 0, (
            f"Make sure to define strictly positive data sampling weights, not {weight}"
        )

        data: DataDir | DataFile
        if Path(path_).is_dir():
            data = DataDir(path=Path(path_))
        elif Path(path_).is_file():
            data = DataFile(path=Path(path_))
        else:
            raise FileNotFoundError(
                f"The path {path_} does not exist. Make sure {path_} is either a file or directory "
                "that contains training data."
            )

        sources.append(data)
        weights.append(weight)

        seen.add(path_)

    sum_weights = sum(weights)
    n_weights = [weight / sum_weights for weight in weights]

    assert min(n_weights) > 0
    assert abs(1 - sum(n_weights)) < 1e-8, (
        f"Defined data sampling weights {weights} must sum to 1."
    )
    return sources, n_weights


def build_dataset(
    args: DataArgs,
    pretrain_data: list[str] | str,
    instruct_tokenizer: InterleavedTokenizer,
    seed: int | None,
    rank: int,
    world_size: int,
    mode: Literal["eval_loss", "mmlu", "swuggy", "sblimp", "ssc", "cd", ""] = "",
    shuffle_pretrain: bool = False,
    is_finite: bool = False
) -> Iterator[Sample]:
    sources, probabilities = parse_data_sources(pretrain_data=pretrain_data)

    shuffle = not mode == "eval_loss" and shuffle_pretrain

    dataset_iterators = [
        get_dataset_iterator(
            args,
            source,
            instruct_tokenizer=instruct_tokenizer,
            rank=rank,
            world_size=world_size,
            is_finite=is_finite,
            seed=seed,
            shuffle_at_epoch=shuffle,
            mode=mode,
        )
        for source in sources
    ]

    if not mode == "":
        combined_iterator = itertools.chain.from_iterable(dataset_iterators)
    else:
        # make sure random_seed is different per rank and original seed
        random_seed = np.array((seed, rank))
        rng = np.random.RandomState(seed=random_seed)
        combined_iterator = interleave_iterators(
            dataset_iterators, probabilities=probabilities, rng=rng
        )

    return combined_iterator


def get_rng(seed: int, rank: int) -> np.random.RandomState:
    random_seed = np.array((seed, rank))
    rng = np.random.RandomState(seed=random_seed)
    return rng

def make_group_shuffled_jsonl(src_jsonl_path, group_size=2, seed=None):
    rng = random.Random(seed)

    with open(src_jsonl_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    groups = [
        lines[i : i + group_size]
        for i in range(0, len(lines) - len(lines) % group_size, group_size)
    ]
    rng.shuffle(groups)
    shuffled_lines = [line for g in groups for line in g]
    if len(lines) % group_size:
        shuffled_lines.extend(lines[-(len(lines) % group_size):])

    # create temp file in same dir
    dirname = os.path.dirname(src_jsonl_path)
    tmp = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        delete=False,
        dir=dirname,
        suffix=".jsonl",
    )
    tmp.writelines(shuffled_lines)
    tmp.close()

    atexit.register(lambda p=tmp.name: os.remove(p) if os.path.exists(p) else None)
    return tmp.name


def get_dataset_iterator(
    args: DataArgs,
    source: DataDir | DataFile,
    instruct_tokenizer: InterleavedTokenizer,
    rank: int,
    world_size: int,
    is_finite: bool,
    seed: int | None,
    shuffle_at_epoch: bool,
    mode: Literal["eval_loss", "mmlu", "swuggy", "sblimp", "ssc", "cd", ""] = "",
) -> Iterator[Sample]:
    epoch = 1
    is_train = mode == ""
    is_mmlu = mode == "mmlu"
    is_paired = mode in ["swuggy", "sblimp", "ssc", "cd"]
    wave_augmenter = WaveAugmenter(args) if is_train else None
    code_augmenter = CodeAugmenter(args) if is_train else None
    while True:
        for jsonl_file in source.jsonl_files:
            if shuffle_at_epoch and is_mmlu:
                jsonl_file = make_group_shuffled_jsonl(jsonl_file, group_size=4, seed=seed)
                seed += 1
            elif shuffle_at_epoch and is_paired:
                jsonl_file = make_group_shuffled_jsonl(jsonl_file, group_size=2, seed=seed)
                seed += 1

            dataset = sphn.dataset_jsonl(
                str(jsonl_file),
                duration_sec=instruct_tokenizer.duration_sec,
                num_threads=4,
                sample_rate=instruct_tokenizer.mimi.sample_rate,
                pad_last_segment=True,
            )
            if shuffle_at_epoch and not (is_mmlu or is_paired):
                dataset = dataset.shuffle(
                    with_replacement=False, skip=rank, step_by=world_size, seed=seed
                )
                seed += 1
            else:
                dataset = dataset.seq(skip=rank, step_by=world_size)

            extra_dict = {}
            if is_mmlu or is_paired:
                extra_paths = list(source.path.rglob("*txt"))
                assert len(extra_paths) > 0, (
                    f"{source.path} does not seem to have any files ending with '.txt'"
                )
                extra_path = extra_paths[0]
                with open(extra_path, "r") as f:
                    extra_dict = {
                        str(entry["file_index"]): entry
                        for entry in (json.loads(line) for line in f if line.strip())
                    }

            for i, sample in enumerate(dataset):
                wav = sample["data"][..., : sample["unpadded_len"]]
                if wave_augmenter:
                    wav = wave_augmenter(wav, instruct_tokenizer.mimi.sample_rate)
                parsed_sample = instruct_tokenizer(wav, sample["start_time_sec"], sample["path"])
                extra = extra_dict.get(str(sample['file_index']))
                if extra:
                    parsed_sample.extra = {}
                    if 'number' in extra:
                        parsed_sample.extra['number'] = extra['number']
                    if 'subject_index' in extra:
                        parsed_sample.extra['subject_index'] = extra['subject_index']
                if code_augmenter:
                    parsed_sample.codes = code_augmenter(parsed_sample.codes)
                if is_mmlu or is_paired or True:
                    length_s = wav.shape[-1] / instruct_tokenizer.mimi.sample_rate
                    length_tokens =  int(np.ceil(length_s * 12.5))
                    parsed_sample.codes = parsed_sample.codes[..., :length_tokens]
                yield parsed_sample
        if is_finite:
            break
        print(f"Rank {rank} finished epoch {epoch}")
        epoch += 1


def interleave_iterators(iterators: list[Iterator], probabilities, rng):
    while True:
        it_id = rng.choice(range(len(iterators)), p=probabilities)
        yield next(iterators[it_id])
