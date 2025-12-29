from typing import Any, Iterator, Literal

from .args import DataArgs
from .dataset import build_dataset
from .interleaver import Batch


def build_data_loader(
    instruct_tokenizer: Any,
    args: DataArgs,
    batch_size: int,
    seed: int | None,
    rank: int,
    world_size: int,
    mode: Literal["eval_loss", "mmlu", "swuggy", "sblimp", "ssc", "cd", ""] = "",
    is_finite: bool = False,
    pretrain_data: str | None = None,
) -> Iterator[Batch]:
    match mode:
        case "eval_loss":
            assert args.eval_loss_data != "", "No eval data provided."
            pretrain_data = args.eval_loss_data
        case "mmlu":
            assert args.mmlu_data != "", "No MMLU data provided."
            pretrain_data = args.mmlu_data
        case "swuggy":
            assert args.swuggy_data != "", "No SLM data provided."
            pretrain_data = args.swuggy_data
        case "sblimp":
            assert args.sblimp_data != "", "No SLM data provided."
            pretrain_data = args.sblimp_data
        case "ssc":
            assert args.ssc_data != "", "No SLM data provided."
            pretrain_data = args.ssc_data
        case _:
            pass

    if pretrain_data is None:
        pretrain_data = args.train_data

    shuffle = args.shuffle
    if mode in ["swuggy", "sblimp", "ssc", "mmlu", "cd"]:
        shuffle = False # Do not shuffle the paired eval datasets.
        
    dataset = build_dataset(
        args=args,
        pretrain_data=pretrain_data,
        instruct_tokenizer=instruct_tokenizer,
        seed=seed,
        rank=rank,
        world_size=world_size,
        mode=mode,
        shuffle_pretrain=shuffle,
        is_finite=is_finite
    )

    sample_list = []
    for sample in dataset:
        assert sample.codes.dim() == 3
        assert len(sample.codes) == 1
        sample_list.append(sample)

        if len(sample_list) == batch_size:
            yield Batch.collate(sample_list)
            sample_list = []
