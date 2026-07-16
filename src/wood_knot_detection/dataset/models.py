from dataclasses import dataclass
from pathlib import Path

@dataclass(frozen=True)
class BoundingBox:
    """
    Bounding box in YOLO normalized coordinates.
    """
    class_id: int
    center_x: float
    center_y: float
    width: float
    height: float

@dataclass(frozen=True)
class Sample:
    """
    One annotated image.
    """
    image_path: Path
    label_path: Path
    board_index: int
    frame_number: int
    annotations: list[BoundingBox]