# CBIS-DDSM (Curated Breast Imaging Subset of DDSM)

This directory contains the **CBIS-DDSM mammography dataset** and its structured subsets for deep learning research.  
It is divided into **calcification** and **mass** lesion categories, along with metadata and raw image storage.

---

##  Directory Structure
```
CBIS-DDSM/
│
├── calc/ # Experiments and models for calcification subset
│ ├── data/ # Processed calcification dataset CSVs
│ ├── models/ # Trained classifier checkpoints (EfficientNet, ConvNeXt, etc.)
│ ├── notebooks/ # Training and evaluation notebooks
│ └── YOLO model/ # YOLO lesion detection for calcifications
│
├── mass/ # Experiments and models for mass subset
│ ├── data/ # Processed mass dataset CSVs
│ ├── models/ # Trained mass lesion classifiers
│ ├── notebooks/ # Training/evaluation notebooks (mass-specific)
│ └── YOLO_mass/ # YOLO lesion detection for masses
│
├── csv/ # Metadata CSVs
│ ├── calc_case_description_train_set.csv
│ ├── calc_case_description_test_set.csv
│ ├── mass_case_description_train_set.csv
│ └── mass_case_description_test_set.csv
│ (Contains lesion-level metadata, labels, BI-RADS, pathology, etc.)
│
├── data/ # Full dataset storage (CBIS-DDSM distributed data)
│ ├── CBIS-DDSM-CALC/ # Calcification cases
│ └── CBIS-DDSM-MASS/ # Mass lesion cases
│
├── jpeg/ # Converted mammography images (from DICOM → JPEG/PNG)
│ ├── calc/ # Calcification images
│ └── mass/ # Mass images

```

---

##  Features

- **Two lesion subsets**:
  - `calc`: Calcifications (micro- & macro-calcification cases).
  - `mass`: Mass lesions (benign vs malignant).
- **Metadata CSVs**:
  - Patient ID, laterality (left/right breast), view (CC/MLO), BI-RADS, pathology confirmation.
- **Image storage**:
  - Original data (`data/`) in DICOM.
  - Preprocessed (`jpeg/`) for model training.
- **Deep Learning pipelines**:
  - YOLO lesion detection → ROI crops.
  - Classifier models trained on cropped patches.
  - Multiple architectures tested (EfficientNet, ResNet, ConvNeXt, BiT, MaxViT).

---

