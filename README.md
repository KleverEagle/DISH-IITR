# DISH Nucleus & Signal Detection

A YOLO11-based segmentation project for detecting and segmenting cell nuclei and gene signals (HER2, Chr17, and fusion) in Dual In Situ Hybridization (DISH) microscopy images.

## Features

- **Nucleus Segmentation**: Identifies and segments cell nuclei.
- **Signal Detection**: Detects and classifies HER2 (black), Chr17 (pink), fusion, and cluster signals.
- **Analysis & Visualization**: Counts signals per image and visualizes prediction overlays.

---

## Setup

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Ensure `yolo11s-seg.pt` (base model) or `best.pt` (trained weights) is in the root directory.

---

## Dataset Structure

> **Note**: This repository **does not include images or labels**. Only the required directory layout and configuration files are provided.

Place your dataset into the following directory structure before training or testing:

```text
data/
├── data.yaml         # YOLO dataset config
├── train.txt         # List of training image paths
├── val.txt           # List of validation image paths
├── train/
│   ├── images/       # Training images (.png, .jpg)
│   └── labels/       # YOLO polygon/box labels (.txt)
└── test/
    ├── images/       # Test images (.png, .jpg)
    └── labels/       # Test labels (.txt)
```

---

## Project Structure

```text
├── code/
│   ├── train.py          # Train YOLO11 segmentation model
│   ├── eval.py           # Evaluate model performance
│   ├── test_and_map.py   # Run inference and count nuclei & signals
│   └── utils/
│       ├── convert.py    # Merge CVAT detection and segmentation labels
│       └── draw_new.py   # Draw predicted masks and signal markers on images
├── data/                 # Dataset folder (see Dataset Structure above)
├── requirements.txt      # Project dependencies
└── README.md
```

---

## Usage

### 1. Training
Train the model with the dataset defined in `data.yaml`:
```bash
python code/train.py
```

### 2. Evaluation
Validate trained weights (`best.pt`):
```bash
python code/eval.py
```

### 3. Inference & Counting
Extract nuclei and signal counts from an image:
```bash
python code/test_and_map.py
```

### 4. Visualization
Generate an annotated image with segmentation masks and detection markers:
```bash
python code/utils/draw_new.py
```
