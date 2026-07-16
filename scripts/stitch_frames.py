'''
Validate that each individual board does not have gaps or overlaps when all frames are joined in one image.

Conclusion: Two consecutive frames have 50% overlap. All frames per individual board have the same image height. All frames (except the last one) have width of 640 px.
'''

import yaml
import cv2

from pathlib import Path
from collections import defaultdict

from typing import Tuple, Dict, List

def load_config(path: Path) -> Dict:
    with open(path, 'r') as file:
        return yaml.safe_load(file)

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

def group_frames_by_board(directory: Path) -> Dict[int, List[Path]]:
    """
    Construct a dictionary where the keys are {board_index}, and the values are a list of image paths belonging to that individual board.

    Args:
        directory (Path): Dataset image directory.

    Returns:
        Dict[int, List[Path]]: Grouped filenames per board.
    """
    boards_dict = defaultdict(list)
    
    for image_path in sorted(directory.glob('*.png'), key=parse_frame_name):
        board_index, _ = parse_frame_name(image_path)
        
        boards_dict[board_index].append(image_path)
    
    return boards_dict

def stitch_board(frames: List[Path]) -> cv2.typing.MatLike:
    """
    Concantenate all board index images ordinally to create a full individual board image.

    Args:
        frames (List[Path]): List of individual board frames.

    Returns:
        cv2.typing.MatLike: Concatanated board image.
    """
    
    images = []
    for frame in frames:
        image = cv2.imread(str(frame), cv2.IMREAD_COLOR)
        images.append(image)
    
    heights = {image.shape[0] for image in images}
    
    # if len(heights) != 1, pad images with black on the bottom
    
    board = cv2.hconcat(images)
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

if __name__ == '__main__':
    config = load_config(Path('configs/config.yaml'))
    image_dir = Path(config.get('dataset').get('images'))
    output_dir = (Path(config.get('output').get('directory'))/'stitched_boards')
    boards = group_frames_by_board(image_dir)
    
    print(f'Found {len(boards)} individual boards.')
    
    for board_index, frames in boards.items():
        
        print(f'Board {board_index} has {len(frames)} frames.')
        
        stitched_board = stitch_board(frames)
        
        save_board_image(output_dir, board_index, stitched_board)