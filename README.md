# wood-knot-detection
Detection of wood knots from sequential board scans.

# Installation
`pip install -r requirements.txt`

# Training
`python -m src.wood_knot_detection.train --seed 42`

`--seed [int]`: Unique seed to generate dataset split.

Saves results in the `runs/` directory.

# Prediction
`python -m src.wood_knot_detection.predict --seed 100 --run train-2`

`--seed [int]`: One of the seeds used to train the model.

`--train-2 [str]`: Which run to run the prediction on.

Generates output in `output/seed_{seed}/{run}` containing generated images and a summary file.

# Prediction on custom test
`python -m src.wood_knot_detection.predict --seed 100 --run train-2 --test_dir data/test_dir`

`--seed [int]`: One of the seeds used to train the model.

`--train-2 [str]`: Which run to run the prediction on.

`--test_dir [str]`: Absolute path to the directory containing `images/` and `labels/`.

Generates `evaluation/` dir inside `test_dir` containing generated images and a summary file.