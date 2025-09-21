# config.py
from dataclasses import dataclass

"""
Config objects for model file locations and detector thresholds.

Purpose:
    I centralize all file paths and detection hyperparameters here so the rest
    of the codebase can import a single source of truth:
      - DetectionConfig: threshold knobs for YOLO-style detection.
      - ModelPaths: absolute/relative paths to my trained models and metadata.

Notes:
    * I keep the dataclasses frozen=True to prevent accidental mutation at runtime. 
"""

@dataclass(frozen=True)
class DetectionConfig:
    """
    Detection thresholds used by the mammography detector.

    THRESH   : Post-processing classification threshold used by my app logic
               (often the operating threshold at which I report a "positive").
    CONF_THR : Detector confidence threshold (objectness/probability cutoff).
    IOU_THR  : IoU threshold for NMS (controls how aggressively boxes are merged).
    """
    THRESH: float = 0.5
    CONF_THR: float = 0.25
    IOU_THR: float = 0.45


@dataclass(frozen=True)
class ModelPaths:
    """
    Canonical file paths to all trained models and their preprocessing metadata.

    Mammography (Stage 1):
      - CALC_* : Calcification detector/classifier weights.
      - MASS_* : Mass detector/classifier weights (placeholders until I add the actual files).

    Early Detection (tabular, dietary/lifestyle proxy):
      - ED_*   : RandomForest model + its required preprocessing artifacts.

    Stage 2 — Coimbra (tabular Keras):
      - COIMBRA_* : Keras model + reference columns + scaling info + target name.

    Stage 3 — Wisconsin (XGBoost):
      - WISC_* : XGBoost model + preprocessing artifacts.

    Conventions:
      * *_REF_COLUMNS  -> list of input features used during training (order matters).
      * *_SCALING_INFO -> serialized scaler params (min/max/mean/std as applicable).
      * *_TARGET_NAME  -> the target column name used at train time (for consistency checks).
    """

    # --- existing: Calcification models ---
    # YOLO detector for calcifications (absolute Windows path to my best weights)
    CALC_YOLO: str = r"C:\Users\PC\Desktop\final project\dashboard\models\cbis-ddsm\calc\yolo_best_model.pt"
    # MaxViT Tiny classifier trained on CBIS-DDSM crops
    CALC_MAXV: str = r"C:\Users\PC\Desktop\final project\dashboard\models\cbis-ddsm\calc\maxvit_tiny_cbis.pt"

    # Mass models 
    MASS_YOLO: str = r"C:\Users\PC\Desktop\final project\dashboard\models\cbis-ddsm\mass\yolo_best_model.pt"
    MASS_MAXV: str = r"C:\Users\PC\Desktop\final project\dashboard\models\cbis-ddsm\mass\maxvit_tiny_cbis.pt"

    # --- NEW: Early Detection (tabular RF) ---
    # RandomForest-based risk model + its preprocessing artifacts
    ED_REF_COLUMNS: str = r"models\colorectal\ref_columns.json"
    ED_SCALING_INFO: str = r"models\colorectal\scaling_info.json"
    ED_TARGET_NAME:  str = r"models\colorectal\target_name.txt"
    ED_MODEL_PATH:   str = r"models\colorectal\rf_model.joblib"

    # --- NEW: Coimbra (tabular Keras) ---
    # Binary classifier + metadata required to faithfully reproduce inference pipeline
    COIMBRA_MODEL:        str = r"models\coimbra\coimbra.keras"
    COIMBRA_REF_COLUMNS:  str = r"models\coimbra\ref_columns.json"
    COIMBRA_SCALING_INFO: str = r"models\coimbra\scaling_info.json"
    COIMBRA_TARGET_NAME:  str = r"models\coimbra\target_name.txt"

    # --- Wisconsin (XGBoost) ---
    # Trained XGBoost model (JSON export) + preprocessing artifacts
    WISC_MODEL:        str = r"models\wisconsin\wisconsin.xgb.json"
    WISC_REF_COLUMNS:  str = r"models\wisconsin\ref_columns.json"
    WISC_SCALING_INFO: str = r"models\wisconsin\scaling_info.json"
    WISC_TARGET_NAME:  str = r"models\wisconsin\target_name.txt"


# Export singletons so the rest of the app can do: from config import CFG, PATHS
CFG = DetectionConfig()
PATHS = ModelPaths()
