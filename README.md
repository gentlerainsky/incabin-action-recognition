# Uniscope 3D

----------------------------------------

## Structures

- `modules/`
  - `definition.py` - specified pose annotation configuration.
  - `pose_extractor/`
    - `pose2d_estimator/` - Implementation of 2D pose estimator using Faster R-CNN, and HR-Net from [MMDet](https://mmdetection.readthedocs.io/) and [MMPose](https://mmpose.readthedocs.io/en/latest/).
    - `pose3d_estimator/` - Implementation of 2D-to-3D Pose Lifter from using JointFormer ([Lutz et al. [2022]](https://github.com/seblutz/JointFormer)).
  - `action_recognizer/`
    - `dataset/`
      - `pose_dataset.py` - Implementation of pose dataset for action recognizer models.
    - `models/`
      - `action_transformer/` - Implementation of Action Transformer ([Mazzia et al. [2021]](https://github.com/PIC4SeR/AcT)).
      - `pose_transformer/` - Implementtation of Pose Transformer (modifiled from [Dominik's repo](https://gitlab.tuwien.ac.at/e193-01-computer-vision/gruppe-gelautz/test02-syntheticcabin/actionrecognition-dominik/-/blob/raquel-dev/models/pose_transformer.py?ref_type=heads)).
      - `stgcn/` - Implementation of ST-GCN ([Yan et al. [2018]](https://github.com/yysijie/st-gcn))
      - `STTFormer/` - Implementation of STTFormer([Qiu et al. [2022]](https://github.com/heleiqiu/STTFormer))
- `demo/` - Data Exploration, and a sample experiement script.
- `experiments/`
  - `experiment.py` - A script to run all experiments.
  - `*_experiment_config.py` - Configs for each experiment for each models.
- `scripts/` - Scripts for data preprocessing including 2D and 3D poses extractor and feature engineering.
- `saved_models/`
  - `pose3d_estimator/` - saved weight of 2D-to-3D lifter models (JointFormer).
  - `action_recognizer/` - saved weight of action recognition models from the experiments.
- `output/` - processed data from preprocessing scripts in `scripts`.

## Installation

- Packages used are listed in `Dockerfile`.
- To use Docker images, we can run the command in `Makefile`.

```bash
# To build the image.
make build
# To run the shell of the image. Don't forget to change the `-v` option to suit the running environment.
make shell
# To run jupyter server
make jupyter
# or
make jupyter-bg
```

## Preprocess

To preprocess Drive&Act. Run scripts in the following order.

1. `scripts/pose2d_extractor.ipynb` - Extract 2D poses from video.
2. `scripts/pose3d_estimator.ipynb` - Perform inference on the 2D poses to get 3D poses.
3. `scripts/pose3d_feature_engineering.ipynb` - Calculate bone length and velocity.

## Run

Experiment can be run with `experiments/experiment.py`
