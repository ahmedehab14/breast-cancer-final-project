# Multimodal Cancer Diagnostic Framework

This project provides an **end-to-end diagnostic system** for cancer risk detection and classification using **imaging (mammography ROI crops)** and **tabular datasets (biochemical, lifestyle, cell-level features)**.  
It integrates multiple models into a **Streamlit-powered dashboard** for clinical interpretation.

---

##  Project Structure

project/
│
├── Breast Cancer Coimbra/ # Biochemical markers dataset & model
│
├── Breast Cancer Wisconsin/ # Cell-level features dataset & XGBoost models
│
├── CBIS-DDSM/ # Mammography dataset (calcification & mass)
│ ├── calc/ # YOLO cropping + classifiers for calcifications
│ ├── mass/ # YOLO cropping + classifiers for masses
│ ├── csv/ # Metadata CSVs (train/test case descriptions)
│ ├── data/ # Raw DICOM data (CBIS-DDSM subset)
│ └── jpeg/ # Preprocessed JPEG images
│
├── colorectal direty and lifestyle/ # Lifestyle & dietary dataset + Random Forest
│
├── dashboard/ # Streamlit diagnostic dashboard
│ ├── pages/ # Multipage dashboard (early detection, diagnosis)
│ ├── models/ # Integrated models (Coimbra, Wisconsin, CBIS, CRC)
│ ├── config.py # Global configuration
│ ├── home.py # Dashboard entrypoint
│ └── ...
│
└── .ipynb_checkpoints/ # Jupyter checkpoints


##  Features

- **Imaging Branch (CBIS-DDSM)**:
  - YOLO lesion detection for **calcifications** and **masses**.
  - ROI crops (+20% context) used for downstream classification.
  - Multiple architectures: ResNet50, EfficientNetV2, ConvNeXtV2, BiT, MaxViT.
  - Explainability with **Grad-CAM++**, **LIME**, and **SHAP**.

- **Tabular Branches**:
  - **Coimbra dataset**: Biochemical markers (Random Forest, XGBoost).
  - **Wisconsin dataset**: Cell-level cytology features (XGBoost).
  - **Colorectal dataset**: Lifestyle/dietary attributes (Random Forest).

- **Fusion Strategy**:
  - Probability-level fusion across imaging + tabular datasets.
  - Methods: **Average**, **Max**, **Noisy-OR**.

- **Dashboard**:
  - Built in **Streamlit**.
  - Upload interface for images & CSVs.
  - Live YOLO overlays + Grad-CAM++/SHAP heatmaps.
  - Metrics: ROC, PR, confusion matrix.
  - Threshold and fusion selector.
  - Exportable PDF reports.

---

##  Datasets

| Dataset    | Type        | Description                              | Location                          |
|------------|-------------|------------------------------------------|-----------------------------------|
| CBIS-DDSM  | Imaging     | Mammography (calcifications, masses)     | `CBIS-DDSM/`                      |
| Coimbra    | Tabular     | Biochemical markers for breast cancer    | `Breast Cancer Coimbra/`          |
| Wisconsin  | Tabular     | Cytological features (benign vs malignant)| `Breast Cancer Wisconsin/`        |
| Colorectal | Tabular     | Dietary & lifestyle risk factors         | `colorectal direty and lifestyle/`|
