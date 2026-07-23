import argparse
import sys
import yaml

from pathlib import Path
from typing import Tuple, Dict, List

from src.wood_knot_detection.dataset.loader import WoodKnotDataset
from src.wood_knot_detection.dataset.splitter import (
    create_dataset_split,
    export_dataset_split,
)
from src.wood_knot_detection.detection.trainer import YOLOTrainer
from src.wood_knot_detection.detection.device import get_device


def load_yaml(path: Path) -> Dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


if __name__ == "__main__":
    # Add seed argument
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed", type=int, default=None, help="Override the seed in config.yaml"
    )
    args = parser.parse_args()

    # Load config.yaml
    config = load_yaml(Path("configs/config.yaml"))

    # Load dataset
    image_dir = Path(config.get("dataset").get("images"))
    label_dir = Path(config.get("dataset").get("labels"))
    dataset = WoodKnotDataset(
        image_dir=image_dir,
        label_dir=label_dir,
    )

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]

    print("First sample:")
    print(f"- Image: {sample.image_path}")
    print(f"- Label: {sample.label_path}")
    print(f"- Board index: {sample.board_index}")
    print(f"- Frame number: {sample.frame_number}")

    print("- Annotations:")
    for annotation in sample.annotations:
        print(annotation)

    # Load splits
    splits_dir = Path(config.get("splits").get("directory"))
    train_ratio = float(config.get("splits").get("train_ratio"))
    val_ratio = float(config.get("splits").get("validation_ratio"))
    test_ratio = float(config.get("splits").get("test_ratio"))
    seed = args.seed if args.seed is not None else int(config.get("splits").get("seed"))
    # Create Splits dataset
    dataset_split = create_dataset_split(
        samples=dataset.samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )
    split_directory = export_dataset_split(
        dataset_split=dataset_split, output_directory=splits_dir
    )

    # dataset.yaml Path
    dataset_yaml = split_directory / "dataset.yaml"

    # Load training.yaml
    training = load_yaml(Path("configs/training.yaml"))

    # Start training
    trainer = YOLOTrainer(
        weights=Path(training.get("model").get("weights")),
        dataset_yaml=dataset_yaml,
        device=get_device(),
    )
    trainer.train(
        epochs=training.get("training").get("epochs"),
        batch_size=training.get("training").get("batch_size"),
        image_size=training.get("training").get("image_size"),
        output_directory=(Path(config.get("output").get("directory")) / f"seed_{seed}"),
    )
