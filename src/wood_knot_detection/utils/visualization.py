import cv2

from typing import Tuple, List

from ..dataset.models import BoundingBox
from .geometry import yolo_to_pixel_coordinates

def draw_bounding_boxes(
    image: cv2.typing.MatLike,
    annotations: List[BoundingBox],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 2,
    show_class_id: bool = True
) -> cv2.typing.MatLike:
    image_height, image_width = image.shape[1]
    
    for annotation in annotations:
        x_min, y_min, x_max, y_max = yolo_to_pixel_coordinates(
            bounding_box=annotation,
            image_width=image_width,
            image_height=image_height
        )
        
        cv2.rectangle(
            img=image,
            pt1=(x_min, y_min),
            pt2=(x_max, y_max),
            color=color,
            thickness=thickness
        )
        
        if show_class_id:
            cv2.putText(
                img=image,
                text=str(annotation.class_id),
                org=(x_min, max(y_min - 5, 15)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=color,
                thickness=1
            )
    
    return image