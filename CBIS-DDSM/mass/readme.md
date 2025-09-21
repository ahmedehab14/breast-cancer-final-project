# Breast Mass Classification (CBIS-DDSM)

This module contains models and resources for **breast mass lesion classification** using the **CBIS-DDSM mass subset**.  
It integrates YOLO-based lesion detection with classifier training on cropped ROIs.

---

##  Directory Structure

Mass/
│
├── data/
│ ├── mass_dataset.csv # Metadata and image paths for mass lesions
│ ├── mass_crops.csv # YOLO-generated ROI crops for masses
│ └── mass_validation_report.csv # Validation report for YOLO crops
│
├── models/
│ ├── efficientnetv2_b4_mass.pt # EfficientNetV2-B4 trained on mass crops
│ ├── resnet50_mass.pt # ResNet50 backbone for mass lesions
│ ├── convnextv2_tiny_mass.pt # ConvNeXtV2-Tiny model
│ └── bit_resnetv2_mass.pt # BiT-ResNetV2 backbone for mass crops
│
├── notebooks/
│ ├── mass_data_exploration.ipynb # EDA on mass dataset
│ ├── mass_training.ipynb # Training scripts for classifiers
│ ├── mass_model_comparison.ipynb # Performance comparison across backbones
│ └── mass_explainability.ipynb # Grad-CAM++/SHAP overlays for lesions
│
├── YOLO_mass/
│ ├── mass.yaml # YOLO dataset config for mass subset
│ ├── YOLO_MASS_TRAIN_DATASET/ # YOLO training dataset
│ ├── MASS_CROPS/ # Cropped ROIs
│ └── YOLO_MASS_PROJECT_output/ # YOLO training outputs (weights, logs)
│
└── README.md


---

##  Features

- **Dataset**: CBIS-DDSM **mass subset** with benign vs malignant labels.
- **YOLO lesion detection**:
  - Crops suspicious **masses** with ±20% padding.
  - Training configs in `mass.yaml`.
- **Classifiers**:
  - Multiple deep learning backbones (EfficientNetV2, ResNet50, ConvNeXtV2, BiT).
  - Trained checkpoints stored in `models/`.
- **Explainability**:
  - Grad-CAM++, SHAP, and LIME overlays available via `mass_explainability.ipynb`.

---


