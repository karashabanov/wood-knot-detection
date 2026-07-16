from pathlib import Path

from dataset.loader import WoodKnotDataset

if __name__ == "__main__":

    image_dir = Path("data/images")
    label_dir = Path("data/labels")

    dataset = WoodKnotDataset(
        image_dir=image_dir,
        label_dir=label_dir,
    )

    print(f"Dataset size: {len(dataset)}")

    sample = dataset[0]

    print("First sample:")
    print(f"Image: {sample.image_path}")
    print(f"Label: {sample.label_path}")
    print(f"Board index: {sample.board_index}")
    print(f"Frame number: {sample.frame_number}")

    print("Annotations:")
    for annotation in sample.annotations:
        print(annotation)