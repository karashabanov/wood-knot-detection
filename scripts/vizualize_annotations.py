"""
Vizualize YOLO annotations - draw the bounding boxes on each image.

Conclusion: Wood knots are correctly labeled. There might be some unlabeled knots (False negatives).
"""

import yaml
import cv2

from pathlib import Path
from typing import Tuple, List


def load_config(path: Path) -> dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)


def load_yolo_labels(label_path: Path) -> List[Tuple[int, float, float, float, float]]:
    """
    Read the labels from the txt files and parse it into a list of tuples.

    Args:
        label_path (Path): Label directory.

    Returns:
        List[Tuple[int, float, float, float, float]]: List of tuples where the elements are: (class_id, center_x_axis, center_y_axis, width, height).
    """
    annotations = []

    if not label_path.exists():
        return annotations

    with open(label_path, "r") as file:
        lines = file.readlines()

        # print(f'number of lines: {len(lines)}')

        for i, line in enumerate(lines):
            # print(i, repr(line))
            annotation = line.strip().strip()

            if not annotation:
                continue

            class_id, cx, cy, width, height = annotation.split(" ")

            annotations.append(
                (int(class_id), float(cx), float(cy), float(width), float(height))
            )

    return annotations


def yolo_2_pixel_coordinates(
    annotation: Tuple[int, float, float, float, float],
    image_width: int,
    image_height: int,
) -> Tuple[int, float, float, float, float]:
    """
    Transform the annotations from yolo-space into pixel space.

    Args:
        annotation (Tuple[int, float, float, float, float]): List of tuples where the elements are: (class_id, center_x_axis, center_y_axis, width, height).
        image_width (int): Image width.
        image_height (int): Image height.

    Returns:
        Tuple[int, float, float, float, float]: List of tuples where the elements are: (class_id, x_min, y_min, x_max, y_max).
    """
    class_id, cx, cy, width, height = annotation

    box_width = width * image_width
    box_height = height * image_height

    center_x = cx * image_width
    center_y = cy * image_height

    x_min = int(center_x - box_width / 2)
    y_min = int(center_y - box_height / 2)

    x_max = int(center_x + box_width / 2)
    y_max = int(center_y + box_height / 2)

    return class_id, x_min, y_min, x_max, y_max


def draw_annotations(
    image: cv2.typing.MatLike, annotations: List[Tuple[int, float, float, float, float]]
) -> cv2.typing.MatLike:
    """
    Draw the bounding box on top of the frame.

    Args:
        image (cv2.typing.MatLike): Input image frame
        annotations (List[Tuple[int, float, float, float, float]]): List of tuples where the elements are: (class_id, x_min, y_min, x_max, y_max).

    Returns:
        cv2.typing.MatLike: Output image frame containing the bounding box.
    """
    height, width = image.shape[:2]
    for annotation in annotations:
        class_id, x_min, y_min, x_max, y_max = yolo_2_pixel_coordinates(
            annotation, width, height
        )

        cv2.rectangle(
            img=image,
            pt1=(x_min, y_min),
            pt2=(x_max, y_max),
            color=(0, 255, 0),
            thickness=2,
        )

        cv2.putText(
            img=image,
            text=f"class {class_id}",
            org=(x_min, max(y_min - 5, 15)),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=0.5,
            color=(0, 255, 0),
            thickness=1,
        )

    return image


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


if __name__ == "__main__":
    config = load_config(Path("configs/config.yaml"))

    image_dir = Path(config["dataset"]["images"])
    label_dir = Path(config["dataset"]["labels"])
    output_dir = Path(config["output"]["directory"]) / "annotated_frames"

    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(image_dir.glob("*.png"), key=parse_frame_name)

    for i, image_path in enumerate(image_paths):
        image = cv2.imread(str(image_path))

        label_path = label_dir / f"{image_path.stem}.txt"
        annotations = load_yolo_labels(label_path=label_path)

        annotated_image = draw_annotations(image, annotations)

        output_path = output_dir / image_path.name
        cv2.imwrite(str(output_path), annotated_image)

        print(f"{i}/{len(image_paths)} index frames completed.")
