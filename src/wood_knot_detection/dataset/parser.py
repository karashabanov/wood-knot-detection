from pathlib import Path

from typing import Tuple, List

from .models import BoundingBox


def parse_frame_name(path: Path) -> Tuple[int, int]:
    """
    Parse image filename from `{board_index}_{frame_number}.png` into a tuple of ({board_index}, {frame_number}).

    Args:
        path (Path): Image path.

    Returns:
        Tuple[int, int]: {board_index}, {frame_number}
    """
    stem = path.stem
    board_index, frame_number = stem.split("_")
    return int(board_index), int(frame_number)


def parse_yolo_labels(label_path: Path) -> List[BoundingBox]:
    """
    Parse YOLO annotation file.

    YOLO format: class_id center_x center_y width height
    Coordinatiates are normalized to [0, 1].

    Args:
        label_path (Path): Labels directory.

    Returns:
        List[BoundingBox]: List of BoundingBox objects.
    """
    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path, "r") as file:
        for line in file:

            annotation = line.strip()

            if not annotation:
                continue

            class_id, cx, cy, width, height = annotation.split(" ")

            annotations.append(
                BoundingBox(
                    class_id=int(class_id),
                    center_x=float(cx),
                    center_y=float(cy),
                    width=float(width),
                    height=float(height),
                )
            )

    return annotations
