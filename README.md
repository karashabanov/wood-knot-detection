# wood-knot-detection
Detection of wood knots from sequential board scans.

# Configuration
The `configs/` directory consists of two yaml files: `config.yaml` and `training.yaml`. 

`config.yaml` handles where the dataset is read from, how to create the data splits and where to save it.

`training.yaml` handles the training hyperparameters and selects which pretrained yolo model to use.

# Local machine

## Installation
`pip install -r requirements.txt`

## Training
`python -m src.wood_knot_detection.train --seed 42`

`--seed [int]`: Unique seed to generate dataset split.

Saves results in the `runs/` directory.

## Prediction
`python -m src.wood_knot_detection.predict --seed 100 --run train-2`

`--seed [int]`: One of the seeds used to train the model.

`--run [str]`: Which run to run the prediction on.

Generates output in `output/seed_{seed}/{run}` containing generated images and a summary file.

## Prediction on custom test
`python -m src.wood_knot_detection.predict --seed 100 --run train-2 --test_dir <path/to/test_dir>`

`--seed [int]`: One of the seeds used to train the model.

`--run [str]`: Which run to run the prediction on.

`--test_dir [str]`: Absolute path to the directory containing `images/` and `labels/`.

Generates `evaluation/` dir inside `test_dir` containing generated images and a summary file.

# Docker
Generated files are written to the mounted project directory.

## Build image
`docker build -t wood-knot-detection .`

## Training
`docker run --rm -v ${PWD}:/app wood-knot-detection python -m src.wood_knot_detection.train --seed 42`

## Prediction
`docker run --rm -v ${PWD}:/app wood-knot-detection python -m src.wood_knot_detection.predict --seed 100 --run train-2`

## Prediction on custom test
`docker run --rm -v <path/to/test_dir>:/app/test_dir wood-knot-detection python -m src.wood_knot_detection.predict --seed 100 --run train-2 --test_dir /app/test_dir`