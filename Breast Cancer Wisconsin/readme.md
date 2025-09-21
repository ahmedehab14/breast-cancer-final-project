# Breast Cancer Wisconsin Dataset & Models

This module implements machine learning models for **breast cancer classification** using the **Wisconsin dataset**.  
It includes the dataset, preprocessing information, trained models (XGBoost format), and a notebook for exploration and evaluation.

---

## Directory Structure
```
Breast Cancer Wisconsin/
│
├── data/
│ └── data.csv # Main dataset (cell-level features)
│
├── models/Wisconsin/
│ ├── metadata.json # Metadata about the model (training setup, params)
│ ├── ref_columns.json # Reference feature columns for input alignment
│ ├── scaling_info.json # Scaling / normalization parameters
│ ├── target_name.txt # Prediction target variable
│ ├── wisconsin.xgb.json # XGBoost model (JSON format)
│ └── wisconsin.xgb.ubj # XGBoost model (binary UBJ format)
│
├── wisconsin.ipynb # Notebook with EDA, training, and evaluation
```

---

##  Features

- **Dataset**: `data.csv` – cell-level features from Wisconsin dataset (benign vs malignant).
- **Preprocessing**:
  - `ref_columns.json`: ensures proper feature alignment.
  - `scaling_info.json`: provides normalization details.
- **Model**:
  - `wisconsin.xgb.json` (human-readable JSON format).
  - `wisconsin.xgb.ubj` (optimized binary format).
- **Metadata**:
  - `metadata.json`: records training parameters and model configuration.
- **Notebook**:
  - `wisconsin.ipynb`: interactive analysis, model training, and performance evaluation.

---
