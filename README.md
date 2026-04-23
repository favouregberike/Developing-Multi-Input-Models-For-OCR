# DigiNsure Inc. Multimodal OCR ID Classification

## Overview

DigiNsure Inc. is modernizing its insurance operations by digitizing historical claim documents. A key challenge in this process is accurately reading and classifying IDs scanned from physical paper documents distinguishing between **primary** and **secondary** IDs across different insurance types.

This project builds a **multimodal deep learning model** that combines:
- **Image data** 64×64 grayscale scans of insurance documents
- **Text data** one-hot encoded insurance type for each document

By fusing these two modalities, the model captures richer contextual information than image recognition alone, improving binary classification accuracy in complex, real-world document scenarios.

---

## Problem Statement

Scanned paper documents introduce noise, inconsistent formatting, and variable image quality that make traditional OCR unreliable on its own. Adding the insurance type as a second input modality allows the model to leverage domain context; for example, the structure of a home insurance document differs significantly from a health or auto insurance document leading to more accurate ID label predictions.

The model predicts one of two labels per image-insurance type pair:
- **Primary ID** = the main identifier on the document
- **Secondary ID** = the supporting identifier on the document

---

## Dataset

The dataset is loaded from `ocr_insurance_dataset.pkl` using a custom `ProjectDataset` class. Each sample contains:

| Field | Description |
|-------|-------------|
| `image` | A 64×64 grayscale scan of an insurance document, shaped `(1, 64, 64)` |
| `insurance_type` | One-hot encoded vector across 5 insurance categories |
| `label` | Binary label — `0` for Primary ID, `1` for Secondary ID |

The dataset is split **80/20** into training and validation sets and loaded via PyTorch `DataLoader` with a batch size of 32.

### Insurance Types

| Label | Insurance Type |
|-------|----------------|
| `home` | Home Insurance |
| `life` | Life Insurance |
| `auto` | Auto Insurance |
| `health` | Health Insurance |
| `other` | Other / Uncategorized |

---

## Model Architecture

The `OCRModel` is a multimodal neural network built with PyTorch (`nn.Module`) consisting of three branches:

### 1. Image Branch (`image_layer`)
Processes the 64×64 grayscale document image:
```
Conv2d(1 → 16, kernel=3, padding=1) → ReLU → MaxPool2d(2×2) → Flatten → Linear(16×32×32 → 128) → ReLU
```

### 2. Insurance Type Branch (`type_layer`)
Processes the one-hot encoded insurance type vector:
```
Linear(5 → 32) → ReLU
```

### 3. Fusion Branch (`fusion_layer`)
Concatenates both feature vectors and classifies:
```
Linear(128 + 32 → 64) → ReLU → Linear(64 → 2)
```

**Total parameters:** printed at runtime via `sum(p.numel() for p in model.parameters())`

---

## Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | Adam (`lr=0.001`) |
| Loss Function | CrossEntropyLoss |
| Epochs | 10 |
| Batch Size | 32 |
| Device | CUDA if available, else CPU |

Training and validation loss and accuracy are tracked across all epochs and visualized in a side-by-side plot at the end of training.

---

## Results

After training, the following metrics are reported:
- **Best Validation Accuracy** across all epochs
- **Final Validation Accuracy** at epoch 10
- **Loss and Accuracy curves** plotted for both training and validation sets

---

## Repository Structure

```
digiNsure-ocr/
│
├── ocr_insurance_dataset.pkl   # Serialized dataset (image + type + label)
├── project_utils.py            # ProjectDataset class definition
├── train.py                    # Full training pipeline (model, training loop, plots)
├── requirements.txt            # Python dependencies
└── README.md
```

## Dependencies

- `torch` ; model definition, training, and inference
- `matplotlib`; training curve visualization
- `numpy` ; image manipulation
- `pickle` ; dataset loading
- `project_utils`; custom `ProjectDataset` class (included in repo)

---

## Business Impact

Automating ID classification from scanned documents will:
- Dramatically reduce manual review time for insurance claims processing
- Improve consistency of primary vs. secondary ID labeling across all insurance types
- Enable DigiNsure to scale document digitization across their full historical archive

---

## Notes

- All document images are anonymized historical records cleared for internal use.
- The model is a proof-of-concept trained on a representative subset of the full archive.
- GPU training is supported automatically when a CUDA-compatible device is available.
