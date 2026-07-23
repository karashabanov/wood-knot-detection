from ..dataset.models import BoundingBox

def calculate_precision(
    true_positives: int,
    false_positives: int
) -> float:
    if true_positives + false_positives == 0:
        return 0.0
    
    return true_positives / (true_positives + false_positives)

def calculate_recall(
    true_positives: int,
    false_negatives: int
) -> float:
    if true_positives + false_negatives == 0:
        return 0.0
    
    return true_positives / (true_positives + false_negatives)

def calculate_f1_score(
    precision: float,
    recall: float
) -> float:
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall) 

def calculate_iou(
    annotation: BoundingBox,
    prediction: BoundingBox
):
    # Annotation corners
    a_x_min = annotation.center_x - annotation.width / 2
    a_y_min = annotation.center_y - annotation.height / 2
    a_x_max = annotation.center_x + annotation.width / 2
    a_y_max = annotation.center_y + annotation.height / 2
    
    # Prediction corners
    p_x_min = prediction.center_x - prediction.width / 2
    p_y_min = prediction.center_y - prediction.height / 2
    p_x_max = prediction.center_x + prediction.width / 2
    p_y_max = prediction.center_y + prediction.height / 2
    
    # Intersection
    intersection_width = max(0.0, min(a_x_max, p_x_max) - max(a_x_min, p_x_min))
    intersection_height = max(0.0, min(a_y_max, p_y_max) - max(a_y_min, p_y_min))
    intersection_area = intersection_width * intersection_height
    
    # Union
    annotation_area = annotation.width * annotation.height
    prediction_area = prediction.width * prediction.height
    union_area = annotation_area + prediction_area - intersection_area
    
    if union_area == 0:
        return 0.0
    
    return intersection_area / union_area
    
    