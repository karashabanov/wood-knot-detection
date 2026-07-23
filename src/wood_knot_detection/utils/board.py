import cv2

from pathlib import Path
from collections import defaultdict
from typing import Tuple, Dict, List

from ..dataset.models import PredictionSample

def parse_frame_name(path: Path) -> Tuple[int, int]:
    """
    Parse image filename from `{board_index}_{frame_number}.png` into a tuple of ({board_index}, {frame_number}).

    Args:
        path (Path): Image path.

    Returns:
        Tuple[int, int]: {board_index}, {frame_number}
    """
    stem = path.stem
    board_index, frame_number = stem.split('_')
    return int(board_index), int(frame_number)

def group_frames_by_board(image_paths: List[Path]) -> Dict[int, List[Path]]:
    """
    Construct a dictionary where the keys are {board_index}, and the values are a list of image paths belonging to that individual board.

    Args:
        directory (List[Path]): List of the Paths (eg. loaded from a manifest).

    Returns:
        Dict[int, List[Path]]: Grouped filenames per board.
    """
    boards_dict = defaultdict(list)
    
    for image_path in sorted(image_paths, key=parse_frame_name):
        board_index, _ = parse_frame_name(image_path)
        
        boards_dict[board_index].append(image_path)
    
    return boards_dict

def group_prediction_samples_by_board(
    prediction_samples: List[PredictionSample]
) -> Dict[int, List[PredictionSample]]:
    boards = defaultdict(list)
    
    for sample in prediction_samples:
        boards[sample.board_index].append(sample)
    
    for samples in boards.values():
        samples.sort(key=lambda sample: sample.frame_number)
    
    return dict(boards)

def stitch_board(frames: List[cv2.typing.MatLike]) -> cv2.typing.MatLike:
    """
    Concantenate all board index images ordinally to create a full individual board image.

    Args:
        frames (List[Path]): List of individual board frames.

    Returns:
        cv2.typing.MatLike: Concatanated board image.
    """
    
    heights = {frame.shape[0] for frame in frames}
    
    # if len(heights) != 1, pad images with black on the bottom
    
    board = cv2.hconcat(frames)
    return board

def save_board_image(output_dir: Path, board_index: int, image: cv2.typing.MatLike) -> None:
    """
    Save the concatanated board image under its index number. 

    Args:
        output_dir (Path): Generated name based on the board index.
        board_index (int): Original board index.
        image (cv2.typing.MatLike): Concatanated board image.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_filename = output_dir / f'board_{board_index}.png'
    
    cv2.imwrite(filename=str(output_filename), img=image)

def clear_board_images(output_dir: Path) -> None:
    """
    Remove previously generated board images from the output directory.

    Args:
        output_dir (Path): Directory containing generated board images.
    """
    for image in output_dir.glob('*.png'):
        image.unlink()