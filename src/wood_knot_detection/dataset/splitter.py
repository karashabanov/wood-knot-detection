import yaml
import json

from dataclasses import dataclass
from collections import defaultdict
from typing import Dict, List, Tuple
from random import Random
from pathlib import Path

from .models import Sample


@dataclass(frozen=True)
class DatasetSplit:
    """
    Custom dataset split.
    """

    seed: int

    train_ratio: float
    val_ratio: float
    test_ratio: float

    train_samples: list[Sample]
    val_samples: list[Sample]
    test_samples: list[Sample]


def group_samples_by_board(samples: List[Sample]) -> Dict[int, List[Sample]]:
    """
    Group dataset samples by board index

    Args:
        samples (List[Sample]): Dataset samples.

    Returns:
        Dict[int, List[Sample]]: Mapped board indices to list of samples (all of its frames.)
    """
    samples_per_board: dict[int, list[Sample]] = defaultdict(list)
    for sample in samples:
        samples_per_board[sample.board_index].append(sample)

    # for board_samples in samples_per_board.values():
    #     board_samples.sort(
    #         key=lambda sample: (
    #             sample.board_index,
    #             sample.frame_number
    #         )
    #     )

    return dict(samples_per_board)


def calculate_split_targets(
    total_samples: int, train_ratio: float, val_ratio: float, test_ratio: float
) -> Tuple[int, int, int]:
    """
    Calculate the number of frames in each split.

    Args:
        total_samples (int): Number of all samples in the raw dataset.
        train_ratio (float): Train ratio.
        val_ratio (float): Valudation ratio.
        test_ratio (float): Test ratio.

    Returns:
        Tuple[int, int, int]: Train target, Validation target, Test target
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum up to 1.")

    train_target = round(total_samples * train_ratio)
    val_target = round(total_samples * val_ratio)
    test_target = round(total_samples * test_ratio)

    return train_target, val_target, test_target


def create_dataset_split(
    samples: List[Sample],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> DatasetSplit:
    """
    Create dataset split. Every board id assigned entirely to one split.
    Boards are shuffled using the provded seed, and then assigned to the split currently having the largest deficit.

    Args:
        samples (List[Sample]): Dataset samples.
        train_ratio (float): Train ratio.
        val_ratio (float): Valudation ratio.
        test_ratio (float): Test ratio.
        seed (int): Seed.

    Returns:
        DatasetSplit: Created custom split.
    """
    grouped_samples = group_samples_by_board(samples)
    board_indices = list(grouped_samples.keys())

    rng = Random(seed)
    rng.shuffle(board_indices)

    train_target, val_target, test_target = calculate_split_targets(
        total_samples=len(samples),
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )

    train_samples: list[Sample] = []
    val_samples: list[Sample] = []
    test_samples: list[Sample] = []

    train_count = 0
    val_count = 0
    test_count = 0

    for board_index in board_indices:

        board_samples = grouped_samples[board_index]
        board_size = len(board_samples)

        deficits = {
            "train": train_target - train_count,
            "val": val_target - val_count,
            "test": test_target - test_count,
        }

        split_name = max(deficits, key=deficits.get)

        if split_name == "train":
            train_samples.extend(board_samples)
            train_count += board_size
        elif split_name == "val":
            val_samples.extend(board_samples)
            val_count += board_size
        else:
            test_samples.extend(board_samples)
            test_count += board_size

    dataset_split = DatasetSplit(
        seed=seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
    )

    return dataset_split


def write_manifest(samples: list[Sample], output_path: Path) -> None:
    with open(output_path, "w") as file:
        for sample in sorted(samples, key=lambda x: x.image_path):
            file.write(f"{sample.image_path.as_posix()}\n")


def write_dataset_yaml(output_path: Path) -> None:
    """
    Write YOLO dataset configuration file.

    Args:
        output_path (Path): Path of dataset.yaml
    """
    dataset_config = {
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt",
        "names": {0: "knot"},
    }

    with open(output_path, "w") as file:
        yaml.safe_dump(dataset_config, file, sort_keys=False)


def write_split_summary(dataset_split: DatasetSplit, output_path: Path) -> None:
    summary = {
        "seed": dataset_split.seed,
        "ratios": {
            "train": dataset_split.train_ratio,
            "val": dataset_split.val_ratio,
            "test": dataset_split.test_ratio,
        },
        "samples": {
            "total": (
                len(dataset_split.train_samples)
                + len(dataset_split.val_samples)
                + len(dataset_split.test_samples)
            ),
            "train": len(dataset_split.train_samples),
            "val": len(dataset_split.val_samples),
            "test": len(dataset_split.test_samples),
        },
        "boards": {
            "train": len(
                {sample.board_index for sample in dataset_split.train_samples}
            ),
            "val": len({sample.board_index for sample in dataset_split.val_samples}),
            "test": len({sample.board_index for sample in dataset_split.test_samples}),
        },
    }

    with open(output_path, "w") as file:
        json.dump(summary, file, indent=4)


def export_dataset_split(dataset_split: DatasetSplit, output_directory: Path) -> Path:
    """
    Export a dataset split into YOLO format.

    Creates:
        output_directory/
            seed_{seed}/
                train.txt
                val.txt
                test.txt
                dataset.yaml
                summary.json

    Args:
        dataset_split (DatasetSplit): Dataset split to export.
        output_directory (Path): Root directory for the exported splits.

    Returns:
        Path: Path to the created split directory.
    """
    split_dir = output_directory / f"seed_{dataset_split.seed}"
    split_dir.mkdir(parents=True, exist_ok=True)

    write_manifest(
        samples=dataset_split.train_samples, output_path=split_dir / "train.txt"
    )
    write_manifest(samples=dataset_split.val_samples, output_path=split_dir / "val.txt")
    write_manifest(
        samples=dataset_split.test_samples, output_path=split_dir / "test.txt"
    )

    write_dataset_yaml(output_path=split_dir / "dataset.yaml")
    write_split_summary(
        dataset_split=dataset_split, output_path=split_dir / "summary.json"
    )

    return split_dir
