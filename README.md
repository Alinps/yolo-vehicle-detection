# YOLO26 Vehicle Detection

A vehicle object detection project built using **YOLO26 (Ultralytics)** and a custom annotated vehicle dataset from Roboflow Universe.

The project uses transfer learning to detect and classify different types of vehicles in images. It predicts object classes, bounding boxes, and confidence scores.

## Project Overview

The objective of this project is to develop an object detection model capable of identifying multiple vehicles in a single image.

The model is trained on a dataset containing 12 vehicle categories. It uses a pretrained YOLO26 Nano model and fine-tunes it on the custom vehicle dataset.

### Key Features

- Multi-class vehicle detection.
- Bounding box localization.
- Confidence score prediction.
- Detection of multiple vehicles in a single image.
- Transfer learning using pretrained YOLO26 weights.
- Model evaluation using precision, recall, and mean Average Precision (mAP).

## Technologies Used

- Python
- Ultralytics YOLO26
- PyTorch
- OpenCV
- NumPy
- Matplotlib
- Jupyter Notebook
- Conda

## Dataset

**Source:** [Roboflow Universe – Vehicles Dataset](https://universe.roboflow.com/roboflow-100/vehicles-q0x2v/dataset/2)

**Version:** 2
**License:** CC BY 4.0
**Task:** Object Detection

The dataset contains 12 vehicle classes:

| Class ID | Vehicle Class |
| -------: | ------------- |
|        0 | big bus       |
|        1 | big truck     |
|        2 | bus-l-        |
|        3 | bus-s-        |
|        4 | car           |
|        5 | mid truck     |
|        6 | small bus     |
|        7 | small truck   |
|        8 | truck-l-      |
|        9 | truck-m-      |
|       10 | truck-s-      |
|       11 | truck-xl-     |

### Dataset Structure

The dataset follows the YOLO object detection annotation format.

```text
data/
└── vehicle-detection/
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    ├── test/
    │   ├── images/
    │   └── labels/
    └── data.yaml
```

Each image has a corresponding `.txt` annotation file containing:

```text
class_id x_center y_center width height
```

The bounding box coordinates are normalized relative to the image dimensions.

The dataset configuration is defined in `data.yaml`.

## Model Architecture

This project uses **YOLO26 Nano (`yolo26n.pt`)**, a pretrained object detection model from Ultralytics.

The pretrained weights are fine-tuned on the custom vehicle dataset using transfer learning.

The `n` in `yolo26n.pt` stands for Nano, which is a lightweight model variant.

## Training Configuration

| Parameter          | Value            |
| ------------------ | ---------------- |
| Model              | YOLO26 Nano      |
| Pretrained weights | `yolo26n.pt`     |
| Training epochs    | 50               |
| Image size         | 640 × 640        |
| Batch size         | 8                |
| Task               | Object Detection |
| Number of classes  | 12               |

### Training Code

```python
from ultralytics import YOLO

# Load pretrained YOLO26 Nano model
model = YOLO("yolo26n.pt")

# Train on the custom vehicle dataset
model.train(
    data="data/vehicle-detection/data.yaml",
    epochs=50,
    imgsz=640,
    batch=8
)
```

The training process generates model checkpoints and evaluation results.

- `best.pt` – Model checkpoint with the best validation performance.
- `last.pt` – Model checkpoint from the final training epoch.

## Model Evaluation

The trained model was evaluated using the validation dataset.

### Evaluation Results

| Metric     |  Score |
| ---------- | -----: |
| Precision  | 0.5125 |
| Recall     | 0.5679 |
| mAP\@50    | 0.4425 |
| mAP\@50-95 | 0.3111 |

### Understanding the Metrics

- **Precision:** Measures how many predicted detections are correct.
- **Recall:** Measures how many actual objects the model successfully detects.
- **mAP\@50:** Mean Average Precision calculated at an IoU threshold of 0.50.
- **mAP\@50-95:** Mean Average Precision averaged across IoU thresholds from 0.50 to 0.95.

These results represent the validation evaluation and are not a separate test-set evaluation.

## Prediction

The trained model can be used to detect vehicles in new images.

### Prediction Code

```python
from ultralytics import YOLO

# Load the trained model
model = YOLO("runs/detect/train/weights/best.pt")

# Run prediction on an external image
results = model.predict(
    source="sample_images/sample_image.jpg",
    conf=0.25,
    save=True
)
```

The model generates predictions with bounding boxes, vehicle class labels, and confidence scores.

## Project Structure

```text
yolo-vehicle-detection/
│
├── data/
│   └── vehicle-detection/
│       └── data.yaml
│
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── predict.py
│
│
├── sample_images/
│   └── sample_image.jpg
│
├── README.md
├── requirements.txt
├── environment.yml
└── .gitignore
```

*The structure above is a suggested organization. Adjust the paths and filenames to match the actual repository.*

## Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Alinps/yolo-vehicle-detection.git

cd yolo-vehicle-detection
```

### 2. Create the Conda Environment

```bash
conda env create -f environment.yml
```

### 3. Activate the Environment

```bash
conda activate yolo_vehicle_detection
```

### 4. Install required Libraries

```bash
pip install requirement.txt
```

### 5. Download the Dataset

Download version 2 of the vehicle dataset from [Roboflow Universe](https://universe.roboflow.com/roboflow-100/vehicles-q0x2v/dataset/2).

Extract the dataset into:

```text
data/vehicle-detection/
```

Ensure that the dataset directory structure matches the paths defined in `data.yaml`.

### 5. Run Training

```bash
python scripts/train.py
```

### 6. Run Prediction

```bash
python scripts/predict.py
```

## Future Improvements

- Improve performance across underrepresented vehicle classes.
- Investigate class imbalance and annotation quality.
- Experiment with data augmentation techniques.
- Evaluate performance on the independent test dataset.
- Compare different YOLO model sizes.
- Improve detection performance in low-light and crowded traffic scenes.
- Explore real-time vehicle detection using video streams.

## License and Dataset Attribution

The vehicle dataset is sourced from Roboflow Universe:

**Vehicles Dataset – Roboflow 100**
[https://universe.roboflow.com/roboflow-100/vehicles-q0x2v/dataset/2](https://universe.roboflow.com/roboflow-100/vehicles-q0x2v/dataset/2)

Dataset version: 2
Dataset license: CC BY 4.0

Dataset attribution and license requirements should be followed when redistributing or using the dataset.

---

**Project:** YOLO26 Vehicle Detection
**Framework:** Ultralytics
**Task:** Multi-class Object Detection
