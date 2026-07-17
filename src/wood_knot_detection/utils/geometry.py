from typing import Tuple

from ..dataset.models import BoundingBox

def yolo_to_pixel_coordinates(
    bounding_box: BoundingBox,
    image_width: int,
    image_height: int
) -> Tuple[int, int, int, int]:
    """
    Transform a YOLO label (bounding box) from normalized coordinates into pixel coordinates.

    Args:
        bounding_box (BoundingBox): Label in YOLO format.
        image_width (int): Image width in pixels.
        image_height (int): Image height in pixels.

    Returns:
        Tuple[int, int, int, int]: (x_min, y_min, x_max, y_max)
    """
    bb_width = bounding_box.width * image_width
    bb_height = bounding_box.height * image_height
    
    center_x = bounding_box.center_x * image_width
    center_y = bounding_box.center_y * image_height
    
    x_min = int(center_x - bb_width / 2)
    y_min = int(center_y - bb_height / 2)
    x_max = int(center_x + bb_width / 2)
    y_max = int(center_y + bb_height / 2)
    
    return (x_min, y_min, x_max, y_max)

def pixel_to_yolo_coordinates(
    x_min_px: int,
    y_min_px: int,
    x_max_px: int,
    y_max_px: int,
    image_width_px: int,
    image_height_px: int,
    class_id: int = 0
) -> BoundingBox:
    """
    Transform a bounding box pixel coordinates into YOLO normalized coordinates.

    Args:
        x_min_px (int): Left edge of the bounding box.
        y_min_px (int): Top edge of the bounding box.
        x_max_px (int): Right edge of the bounding box.
        y_max_px (int): Bottom edge of the bounding box.
        image_width_px (int): Image width in pixels.
        image_height_px (int): Image height in pixels
        class_id (int, optional): Class identifier.

    Returns:
        BoundingBox: Bounding box annotation in YOLO coordinates.
    """
    bb_width_px = x_max_px - x_max_px
    bb_width_norm = bb_width_px / image_width_px
    
    bb_height_px = y_max_px - y_min_px
    bb_height_norm = bb_height_px / image_height_px
    
    center_x_px = x_min_px + bb_width_px / 2
    center_x_norm = center_x_px / image_width_px
    
    center_y_px = y_min_px + bb_height_px / 2
    center_y_norm = center_y_px / image_height_px
    
    return BoundingBox(
        class_id=class_id,
        center_x=center_x_norm,
        center_y=center_y_norm,
        width=bb_width_norm,
        height=bb_height_norm
    )