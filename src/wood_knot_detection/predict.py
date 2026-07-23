import argparse
import yaml
import cv2

from pathlib import Path
from typing import Tuple, Dict, List

from src.wood_knot_detection.dataset.loader import WoodKnotDataset
from src.wood_knot_detection.detection.detector import YOLODetector
from src.wood_knot_detection.evaluation.evaluator import Evaluator
from src.wood_knot_detection.dataset.models import PredictionSample
from src.wood_knot_detection.utils import board
from src.wood_knot_detection.utils.visualization import draw_bounding_boxes

def load_yaml(path: Path) -> Dict:
    with open(path, "r") as file:
        return yaml.safe_load(file)

def load_test_manifest(path: Path) -> List[Path]:
    with open(path, 'r') as file:
        return [
            Path(line.strip())
            for line in file.readlines()
            if line.strip()
        ]

if __name__ == "__main__":
    # Add arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--seed',
        type=int,
        required=True,
        help='Dataset split seed.'
    )
    parser.add_argument(
        '--run',
        type=str,
        required=True,
        help='Training run name containing the trained weights.'
    )
    parser.add_argument(
        '--test_dir',
        type=str,
        required=False,
        help="Custom test dataset containing images/ and labels/."
    )
    args = parser.parse_args()
    
    # Load config.yaml
    config = load_yaml(Path('configs/config.yaml'))
    
    # Load dataset
    if args.test_dir is not None:
        image_dir = Path(args.test_dir) / 'images'
        label_dir = Path(args.test_dir) / 'labels'
    else:
        image_dir = Path(config.get('dataset').get('images'))
        label_dir = Path(config.get('dataset').get('labels'))
    
    dataset = WoodKnotDataset(
        image_dir=image_dir,
        label_dir=label_dir
    )
    
    # LookUp for Ground Truth samples (raw dataset)
    samples_by_image = {
        sample.image_path: sample
        for sample in dataset.samples
    }
    
    # Load test images
    if args.test_dir is not None:
        image_paths = sorted(image_dir.glob('*.png'))
    else:
        # Construct test manifest path
        test_manifest_path = (Path('data') / 'splits' / f'seed_{args.seed}' / 'test.txt')
        image_paths = load_test_manifest(test_manifest_path)
    
    # Constuct model path
    model_path = (Path('runs') / 'detect' / 'output' / f'seed_{args.seed}' / f'{args.run}' / 'weights' / 'best.pt')
    print(f'Loading model: {model_path}')
    
    # Create detector object
    detector = YOLODetector(weights=model_path)
    
    # Run inference
    prediction_samples: List[PredictionSample] = []
    total_detections = 0 # Bounding boxes count, YOLO found
    total_annotations = 0 # Bounding boxes count, Ground Truth
    
    print('Starting inference...')
    for image_path in image_paths:
        # Ground truth
        sample = samples_by_image.get(image_path)
        # Prediction
        prediction = detector.predict(image_path=image_path)
        
        prediction_sample = PredictionSample(
            image_path=sample.image_path,
            label_path=sample.label_path,
            board_index=sample.board_index,
            frame_number=sample.frame_number,
            annotations=sample.annotations,
            detections=prediction.detections
        )
        prediction_samples.append(prediction_sample)
        
        total_detections += len(prediction.detections)
        total_annotations += len(sample.annotations)
    print('Inference finished.')
    
    print('Stitching frames...')
    output_dir = (Path(config.get('output').get('directory'))/f'seed_{args.seed}'/f'{args.run}')
    board.clear_board_images(output_dir=output_dir)
    
    # Draw bounding boxes, stitch and save
    boards = board.group_prediction_samples_by_board(prediction_samples)
    for board_index, board_samples in boards.items():
        
        # Draw bounding boxes (ground truth + predictions) on each frame
        frames_per_board = []
        for sample in board_samples:
            sample.frame_number
            sample.image_path
            sample.annotations
            sample.detections
        
            frame = cv2.imread(str(sample.image_path))
            # Draw annotations
            frame = draw_bounding_boxes(frame, sample.annotations, (0, 255, 0))
            # Draw predictions
            frame = draw_bounding_boxes(frame, sample.detections, (255, 0, 0))
            
            frames_per_board.append(frame)
        
        # Stitch all frames per board
        stitched_board = board.stitch_board(frames_per_board)
        
        board.save_board_image(
            output_dir=output_dir,
            board_index=board_index,
            image=stitched_board
        )
    print('Stithing frames finished.')
    
    print('Evaluating results...')
    # Evaluate results
    evaluation_result = Evaluator.evaluate(prediction_samples, 0.5)
    Evaluator.save(evaluation_result=evaluation_result, output_dir=output_dir)
    print('Evaluation finished.')
    print(f'See results in {output_dir}.')
    