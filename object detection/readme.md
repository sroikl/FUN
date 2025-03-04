# Object Detection and Embedding

This project implements a two-stage pipeline for object detection and embedding:
1. Detect objects in images using pre-trained object detectors (Faster R-CNN or Mask R-CNN)
2. Extract feature embeddings for each detected object using feature extractors (ResNet50 or EfficientNet)

## Features

- Object detection with configurable confidence threshold
- Feature extraction from detected regions of interest (ROIs)
- Visualization of detection results
- Visualization of embeddings using PCA
- Distance calculation between class embeddings

## Project Structure

- `detectors.py`: Defines the detector models (Faster R-CNN, Mask R-CNN)
- `feature_extractors.py`: Defines the feature extraction models (ResNet50, EfficientNet)
- `object_embedder.py`: Main class that combines detection and embedding
- `viz_utils.py`: Visualization utilities for detections and embeddings
- `main_exp.py`: Entry point script to run experiments

## Installation

1. Clone this repository
2. Install dependencies:
```
pip install -r requirements.txt
```

## Usage

Place your images in a directory (default is `data/`), then run:

```bash
python main_exp.py --data_dir data --threshold 0.9 --feature_extractor efficientnet --detector_name fasterrcnn --save_results True
```

### Arguments

- `--data_dir`: Directory containing images (default: 'data')
- `--threshold`: Detection confidence threshold (default: 0.9)
- `--feature_extractor`: Feature extractor model to use (options: 'resnet50', 'efficientnet', default: 'efficientnet')
- `--detector_name`: Object detector to use (options: 'fasterrcnn', 'maskrcnn','yolo, default: 'fasterrcnn')
- `--save_results`: Save the results in a 'results directory'

## Output

Results are saved in the `results/` directory with the following structure:
```
results/
└── image_name_detector_featureextractor/
    ├── detections.png
    ├── embedding_visualization_pca.png
    └── euclidean_distances.csv
```

- `detections.png`: Visualization of detected objects with bounding boxes
- `embedding_visualization_pca.png`: 2D visualization of object embeddings using PCA
- `euclidean_distances.csv`: Euclidean distances between class centers

