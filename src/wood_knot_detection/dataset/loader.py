from pathlib import Path

from .models import Sample
from .parser import parse_frame_name, parse_yolo_labels


class WoodKnotDataset:
    """
    Dataset loader for wood knot images and YOLO annotations.

    Expected data structure:
    data/
        images/
            {board_index}_{frame_number}.png
        labels/
            {board_index}_{frame_number}.txt
    """

    def __init__(self, image_dir: Path, label_dir: Path):
        """
        Initialize dataset

        Args:
            image_dir (Path): Image directory.
            label_dir (Path): Label directory.
        """
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.samples = self._create_samples()

    def _create_samples(self) -> list[Sample]:
        """
        Craete dataset sample objects from raw data.

        Returns:
            list[Sample]: _description_
        """
        samples = []
        image_paths = sorted(self.image_dir.glob("*.png"), key=parse_frame_name)

        for image_path in image_paths:
            board_index, frame_number = parse_frame_name(image_path)

            label_path = self.label_dir / f"{image_path.stem}.txt"
            annotations = parse_yolo_labels(label_path)

            samples.append(
                Sample(
                    image_path=image_path,
                    label_path=label_path,
                    board_index=board_index,
                    frame_number=frame_number,
                    annotations=annotations,
                )
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Sample:
        return self.samples[index]
