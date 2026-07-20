from pathlib import Path
from ultralytics import YOLO

class YOLOTrainer:
    """
    Wrapper class for YOLO training.
    - Loads weights
    - Configures training hyperparameters
    - Starts training
    """
    def __init__(
        self,
        weights: Path,
        dataset_yaml: Path,
        device: str
    ):
        self.weights = weights
        self.dataset_yaml = dataset_yaml
        self.device = device
        self.model = YOLO(str(weights))
    
    def train(
        self,
        epochs: int,
        batch_size: int,
        image_size: int,
        output_directory: Path
    ) -> None:
        self.model.train(
            data=str(self.dataset_yaml),
            epochs=epochs,
            batch=batch_size,
            imgsz=image_size,
            device=self.device,
            project=str(output_directory),
            rect=True,
            auto_augment=None,
            mosaic=0.0,
            translate=0.05,
            scale=0.1,
            fliplr=0.5,
            flipud=0.5,
            hsv_h=0.0,
            hsv_s=0.0,
            hsv_v=0.0,
        )