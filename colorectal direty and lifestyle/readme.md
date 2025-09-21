# Colorectal Cancer Risk (Dietary & Lifestyle)

This module focuses on **early detection of colorectal cancer risk** using dietary and lifestyle data.  
It includes the dataset, preprocessing references, and a trained Random Forest model pipeline.

---

## Directory Structure

colorectal dietry and lifestyle/
│
├── data/
│ └── crc_dataset.csv # Main dataset (dietary & lifestyle features)
│
├── model files/
│ ├── ref_columns.json # Reference feature columns for model input
│ ├── rf_model.joblib # Trained Random Forest model
│ ├── scaling_info.json # Normalization / scaling parameters
│ └── target_name.txt # Name of target variable
│
├── early.ipynb # Notebook with data exploration & model training


---

##  Features

- **Dataset**: `crc_dataset.csv` contains dietary and lifestyle attributes linked to colorectal cancer risk.
- **Preprocessing**:
  - `ref_columns.json` – ensures correct feature ordering.
  - `scaling_info.json` – stores scaling/normalization parameters.
- **Model**: `rf_model.joblib` (Random Forest) trained on processed features.
- **Target**: `target_name.txt` defines the prediction target (e.g., cancer risk label).

---
