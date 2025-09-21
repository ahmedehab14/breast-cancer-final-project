# Breast calc Classification (CBIS-DDSM)

This module focuses on **mammography image analysis** using **YOLO-based lesion cropping** combined with multiple deep learning backbones (CNN/Transformer).  
It includes YOLO preprocessing, trained classifier weights, comparative evaluations, and Jupyter notebooks for experimentation.

---

## Directory Structure

calc/
│
├── data/
│ ├── model_comapre.csv # Metrics summary across models
│ ├── model_comapre_ranked.csv # Ranked comparison of models
│ ├── yolo_crops.csv # YOLO-generated ROI crops (metadata)
│
├── models/ # Pretrained classifier checkpoints
│ ├── bit_resnetv2_101x1_cbis.pt
│ ├── convnextv2_tiny_cbis.pt
│ ├── efficientnetv2_b4_cbis.pt
│ └── maxvit_tiny_cbis.pt
│
├── notebooks/ # Jupyter notebooks for training & analysis
│ ├── BiT-ResNetV2.ipynb
│ ├── ConvNeXtV2.ipynb
│ ├── efficientnetv2b4_training_yolo_crops.ipynb
│ ├── maxvit_tiny_tf_224.ipynb
│ ├── model_comapre.ipynb
│ ├── model_ensemble.ipynb
│ ├── resnet50_training_cbis_crops.ipynb
│ └── resnet50_training_yolo_crop.ipynb
│
├── YOLO model/ # YOLO lesion detection pipeline
│ ├── CBIS_YOLO_PROJECT_output/ # YOLO outputs (trained weights, logs)
│ ├── data/ # Dataset for YOLO training
│ │ ├── CBIS_YOLO_TRAIN_DATASET/ # Training dataset structure
│ │ └── YOLO_CROPS/ # Cropped ROIs
│ ├── YOLO_CROPS_CSV_VIZ/ # Visualization of ROI crops
│ ├── calc_preprocessed_clean.csv # Preprocessed dataset after cleaning
│ ├── cbis.yaml # YOLO dataset config file
│ ├── yolo_crops_validation_report.csv# Validation report
│ ├── yolo_crops.csv # YOLO crops metadata
│
├── models/ # (duplicate folder for alt. model saves)
├── notebooks/ # (duplicate folder for YOLO notebooks)
│ ├── yolo_data_generation.ipynb # Data generation for YOLO
│ ├── YOLO_logs.txt # Training logs
│ └── YOLO_model_train.ipynb # YOLO training notebook
│
└── readme.md # Project documentation


---

## Features

- **YOLO Branch**:
  - Lesion detection on CBIS-DDSM mammograms.
  - ROI cropping with ±20% context.
  - Training configs in `cbis.yaml`.
- **Classifier Branch**:
  - State-of-the-art CNN/Transformer backbones:
    - BiT-ResNetV2
    - ConvNeXtV2
    - EfficientNetV2-B4
    - MaxViT
    - ResNet50
  - Trained checkpoints saved as `.pt`.
- **Evaluation**:
  - `model_comapre.csv` and `model_comapre_ranked.csv` summarize AUROC, F1, sensitivity, specificity, etc.
  - Ensemble experiments available (`model_ensemble.ipynb`).
- **Explainability**:
  - Notebooks include Grad-CAM++ overlays for lesion localization.

---
