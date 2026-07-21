from dataclasses import dataclass
from pathlib import Path
from typing import List

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
    annotations: List[BoundingBox]

@dataclass(frozen=True)
class Detection:
    class_id: int
    confidence: float
    bounding_box: BoundingBox

@dataclass(frozen=True)
class ImagePrediction:
    image_path: Path
    detections: List[Detection]

@dataclass(frozen=True)
class PredictionSample(Sample):
    detections: List[Detection]