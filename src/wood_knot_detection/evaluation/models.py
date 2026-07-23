from dataclasses import dataclass
from typing import List
from ..dataset.models import BoundingBox

@dataclass
class DetectionMatch:
    class_id: int
    confidence: float
    iou: float
    annotation: BoundingBox | None
    detection: BoundingBox | None
    
    is_true_positive: bool
    is_false_positive: bool
    is_false_negative: bool

@dataclass
class EvaluationResult:
    total_samples: int
    total_annotations: int
    total_predictions: int

    true_positives: int
    false_positives: int
    false_negatives: int
    
    precision: float
    recall: float
    f1_score: float
    mean_iou: float
    
    detection_matches: List[DetectionMatch]