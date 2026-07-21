from typing import List
from pathlib import Path
from ultralytics import YOLO

from .device import get_device
from ..dataset.models import BoundingBox, Detection, ImagePrediction

class YOLODetector:
    """
    Wrapper class for YOLO training.
    - Loads weights
    - Configures training hyperparameters
    - Starts training
    """
    def __init__(
        self,
        weights: Path,
        device: str | None = None
    ):
        self.weights = weights
        self.device = device if device is not None else get_device()
        self.model = YOLO(str(weights))
    
    def predict(
        self,
        image_path: Path,
        confidence_threshold: float = 0.25
    ) -> ImagePrediction:
        results = self.model.predict(
            source=str(image_path),
            device=self.device,
            conf=confidence_threshold,
            verbose=False
        )
        
        detections = []
        
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                confidence = float(box.conf.item())
                
                xywh = box.xywhn[0]
                bounding_box = BoundingBox(
                    class_id=class_id,
                    center_x=float(xywh[0]),
                    center_y=float(xywh[1]),
                    width=float(xywh[2]),
                    height=float(xywh[3])
                )
                
                detections.append(
                    Detection(
                        class_id=class_id,
                        confidence=confidence,
                        bounding_box=bounding_box
                    )
                )
        
        return ImagePrediction(
            image_path=image_path,
            detections=detections
        )