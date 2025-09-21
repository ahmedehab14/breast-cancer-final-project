# Diagnostic Dashboard Project

This repository contains an end-to-end framework for **early cancer detection and diagnosis** using both imaging (mammography ROI crops) and tabular datasets (biochemical, cell-level, lifestyle).  
The system integrates multiple models, evaluation scripts, and a Streamlit-powered dashboard for clinical interpretation.

---

##  Project Structure

dashboard/
│
├── models/ # Model scripts organized by dataset
│ ├── cbis-ddsm/ # Mammography (ROI crops from YOLO detection)
│ ├── coimbra/ # Biochemical markers dataset
│ ├── colorectal/ # Lifestyle/dietary features dataset
│ └── wisconsin/ # Cell-level features dataset
│
├── pages/ # Streamlit multipage dashboard
│ ├── 1_Early_Detection.py # Page for preliminary risk screening
│ ├── 2_Full_Diagnosis.py # Page for comprehensive multimodal diagnosis
│ └── samples.ipynb # Notebook exploring dataset sampling
│
├── coimbra_negative_samples.csv
├── coimbra_positive_samples.csv
├── early_detection_crc0_samples.csv
├── early_detection_crc1_samples.csv
├── wisconsin_negative_sample.csv
├── wisconsin_positive_sample.csv
│
├── cbis_ddsm_mammography.py # Model pipeline for CBIS-DDSM images
├── coimbra_model.py # Model pipeline for Coimbra dataset
├── wisconsin_model.py # Model pipeline for Wisconsin dataset
│
├── config.py # Central configuration (paths, params, etc.)
├── home.py # Streamlit entrypoint
└── pycache/ # Python cache files

##  Features

- **Imaging Branch**: ROI-based mammography classifiers (EfficientNet, ConvNeXt, BiT).
- **Tabular Branches**: Coimbra (biochemical), Wisconsin (cell-level), Colorectal (lifestyle).
- **Fusion**: Probability-level combination (Average, Max, Noisy-OR).
- **Dashboard**: Interactive Streamlit app with:
  - Uploads (images + CSVs)
  - Grad-CAM++ / SHAP explainability overlays
  - ROC/PR curves, confusion matrices
  - Threshold and fusion selector
  - Exportable PDF report

---

##  Datasets

| Dataset    | Type        | Contents                          | Samples (CSV)                 
|------------|-------------|-----------------------------------|-------------------------------
| CBIS-DDSM  | Imaging     | Mammography ROI crops + metadata  | `cbis_ddsm_mammography.py`    
| Coimbra    | Tabular     | Biochemical markers               | `coimbra_negative/positive.csv`
| Wisconsin  | Tabular     | Cell-level features               | `wisconsin_negative/positive.csv` 
| Colorectal | Tabular     | Lifestyle/dietary survey          | `early_detection_crc0/1.csv`  

## Run the dashboard:

streamlit run home.py
