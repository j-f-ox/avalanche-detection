# Avalanche Detection

Research code for paper "Automating avalanche detection in ground-based photographs with deep learning". The accompanying dataset of avalanche photographs is available at [https://researchdata.uibk.ac.at//records/h07f4-qzd17](https://researchdata.uibk.ac.at//records/h07f4-qzd17).

This repository contains code to train and evaluate ResNet, VGG, and YOLOv3 models to detect avalanches in photographs.

![!\[Alt text\](<.e.png>)](segmentation-example.png)

## Setup

1. Clone the respository
   ```
   git clone git@github.com:j-f-ox/avalanche-detection.git
   cd avalanche-detection
   ```

2. Set up virtual environment and install dependencies (requires Python 3.9). Note that the PyTorch version might need to be modified on different systems.
   ```
   python3.9 -m venv .venv
   source .venv/bin/activate
   sh setup.sh
   ```

3. Download and unzip the dataset
   ```
   curl -L -o uibk_avalanches.zip https://researchdata.uibk.ac.at//records/h07f4-qzd17/files/uibk_avalanches.zip
   unzip -q -d .data uibk_avalanches.zip
   ```

4. Generate the train/test split
   ```
   python3 utils/train_test_split.py
   ```


## Usage

We support two tasks: [Image Classification](#image-classification) to classify entire images and [Avalanche Segmentation](#avalanche-segmentation) to identify each avalanche in an image.

### Image Classification

The image classification task is to determine whether a photograph contains any visible avalanches and classify the predominant avalanche release mechanism if so. Each image should be assigned a single label out of `loose`, `glide`, `slab`, or `none`.

See scripts in [classification/experiments](classification/experiments) for examples of how to get started training ResNet and VGG models.

### Avalanche Segmentation

The avalanche segmentation task is to detect and label a bounding box for each avalanche region in a photograph. Each bounding box should be assigned a single label out of `loose`, `glide`, or `slab`.

The code in [/segmentation](/segmentation) supports training YOLO models and is based on a clone of the [Ultralytics yolov5 repository](https://github.com/ultralytics/yolov5) made on 2023-03-19. After downloading the dataset (described in [#Setup](#setup)), generate annotations in a format supported by Ultralytics:
```
python3 segmentation/data/scripts/generate_yolo_annotations.py
```

See scripts in [segmentation/experiments](segmentation/experiments) for examples of how to get started training models.

## Contributors

- The code in [/classification](/classification) repository contains code contributions from James Fox, Sandra Reitinger, and David Peer.

- The code in [/segmentation](/segmentation) is based on a clone of https://github.com/ultralytics/yolov5 made on 2023-03-19. Any bugs introduced are our own.
