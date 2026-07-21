import argparse
import yaml

from pathlib import Path
from typing import Tuple, Dict, List

from src.wood_knot_detection.detection.detector import YOLODetector

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
    args = parser.parse_args()
    
    # Load config.yaml
    config = load_yaml(Path('configs/config.yaml'))
    
    # Construct test manifest path
    test_manifest_path = (Path('data') / 'splits' / f'seed_{args.seed}' / 'test.txt')
    image_paths = load_test_manifest(test_manifest_path)
    
    # Constuct model path
    model_path = (Path('runs') / 'detect' / 'output' / f'seed_{args.seed}' / f'{args.run}' / 'weights' / 'best.pt')
    print(f'Loading model: {model_path}')
    
    # Create detector object
    detector = YOLODetector(weights=model_path)
    
    # Run inference
    total_detections = 0
    for image_path in image_paths:
        prediction = detector.predict(image_path=image_path)
        print(f'\nImage: {prediction.image_path}')
        total_detections += len(prediction.detections)
        
        for detection in prediction.detections:
            print(detection)
    
    print(f'Total detections: {total_detections}')