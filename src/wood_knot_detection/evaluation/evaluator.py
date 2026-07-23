import json

from pathlib import Path
from typing import List

from dataclasses import asdict

from ..dataset.models import PredictionSample, BoundingBox, Detection
from .models import EvaluationResult, DetectionMatch
from .metrics import (
    calculate_precision,
    calculate_recall,
    calculate_f1_score,
    calculate_iou
)
from ..dataset.models import Sample, BoundingBox, PredictionSample

class Evaluator:
    @staticmethod
    def evaluate(
        prediction_samples: List[PredictionSample],
        iou_threshold: float = 0.5
    ) -> EvaluationResult:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        total_annotations = 0
        total_predictions = 0
        
        total_true_positive_iou = 0.0
        true_positive_count = 0
        
        all_matches: List[DetectionMatch] = []
        
        for sample in prediction_samples:
            total_annotations += len(sample.annotations)
            total_predictions += len(sample.detections)
            
            _tp, _fp, _fn, detection_matches = Evaluator._match_detections(
                sample.annotations,
                sample.detections,
                iou_threshold
            )
            
            true_positives += _tp
            false_positives += _fp
            false_negatives += _fn
            
            all_matches.extend(detection_matches)
            
            for match in detection_matches:
                if match.is_true_positive:
                    total_true_positive_iou += match.iou
                    true_positive_count += 1
        
        precision = calculate_precision(true_positives, false_positives)
        recall = calculate_recall(true_positives, false_negatives)
        f1 = calculate_f1_score(precision, recall)
        mean_iou = (total_true_positive_iou / true_positive_count if true_positive_count > 0 else 0.0)
        
        return EvaluationResult(
            total_samples=len(prediction_samples),
            total_annotations=total_annotations,
            total_predictions=total_predictions,
            true_positives=true_positives,
            false_positives=false_positives,
            false_negatives=false_negatives,
            precision=precision,
            recall=recall,
            f1_score=f1,
            mean_iou=mean_iou,
            detection_matches=all_matches
        )
        
    @staticmethod
    def _match_detections(
        annotations: List[BoundingBox],
        detections: List[Detection],
        iou_threshold: float
    ) -> tuple[int, int, int, List[DetectionMatch]]:
        true_positives = 0
        false_positives = 0
        false_negatives = 0
        
        matches: List[DetectionMatch] = []
        matched_annotations = set()
        
        for detection in sorted(detections, key=lambda x: x.confidence, reverse=True):
            best_iou = 0.0
            best_annotation_index = None
            
            for index, annotation in enumerate(annotations):
                
                # Skip matched label
                if index in matched_annotations:
                    continue
                
                iou = calculate_iou(annotation, detection.bounding_box)
                if iou > best_iou:
                    best_iou = iou
                    best_annotation_index = index

            if best_annotation_index is not None and best_iou >= iou_threshold:
                true_positives += 1
                matched_annotations.add(best_annotation_index)
                matches.append(DetectionMatch(
                    class_id=detection.class_id,
                    confidence=detection.confidence,
                    iou=best_iou,
                    annotation=annotations[best_annotation_index],
                    detection=detection.bounding_box,
                    is_true_positive=True,
                    is_false_positive=False,
                    is_false_negative=False
                ))
            else:
                false_positives += 1
                matches.append(DetectionMatch(
                    class_id=detection.class_id,
                    confidence=detection.confidence,
                    iou=0.0,
                    annotation=None,
                    detection=detection.bounding_box,
                    is_true_positive=False,
                    is_false_positive=True,
                    is_false_negative=False
                ))
        
        for index, annotation in enumerate(annotations):
            if index not in matched_annotations:
                false_negatives += 1
                matches.append(DetectionMatch(
                    class_id=annotation.class_id,
                    confidence=0.0,
                    iou=0.0,
                    annotation=annotation,
                    detection=None,
                    is_true_positive=False,
                    is_false_positive=False,
                    is_false_negative=True
                ))
        
        return (
            true_positives,
            false_positives,
            false_negatives,
            matches
        )

    @staticmethod
    def save(
        evaluation_result: EvaluationResult,
        output_dir: Path,
    ) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / 'evaluation.json'
        
        with open(output_path, 'w', encoding='utf-8') as file:
            json.dump(
                asdict(evaluation_result),
                file,
                indent=4
            )