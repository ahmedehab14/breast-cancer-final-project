# wisconsin_model.py
import json, os
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# I load Keras lazily to avoid hard dependency errors when only XGBoost is needed.
try:
    from tensorflow.keras.models import load_model as _keras_load
except Exception:
    _keras_load = None

# Same idea for XGBoost—optional import with a graceful fallback.
try:
    import xgboost as xgb
except Exception:
    xgb = None


def _resolve(p: str) -> str:
    """
    Resolve a user-provided path to an absolute path with a couple of sensible fallbacks.

    Order I try:
      1) Treat p as given. If it exists, return its absolute path.
      2) Treat p as relative to this file's directory (useful when I ship models with the code).
      3) Return the absolute version of p anyway (helps produce clearer error messages).

    I keep this helper small and deterministic because file discovery bugs are painful.
    """
    p1 = Path(p)
    if p1.exists():
        return str(p1.resolve())
    p2 = Path(__file__).resolve().parent / p1
    if p2.exists():
        return str(p2.resolve())
    return str(p1.resolve())  # absolute for error msgs


class WisconsinTabularModel:
    """
    Unified tabular model loader for the Wisconsin dataset pipeline.

    What it supports:
      - Keras models exported as .keras / .h5 (binary classifier with a single sigmoid output).
      - XGBoost models exported as .json / .ubj / .model (Booster).

    Preprocessing:
      - Min–max scaling using per-feature min/max values extracted from training.
      - Column order is locked to `ref_columns` to prevent leakage / mismatch.

    Inputs:
      model_path        : Path to Keras (.keras/.h5) or XGBoost (.json/.ubj/.model).
      ref_columns_path  : JSON list of feature names used in training (strict order).
      scaling_info_path : JSON mapping feature -> {"min": ..., "max": ...}.
      target_name_path  : (Optional) text file containing the target column name (for sanity checks/UI only).

    I deliberately keep this class framework-agnostic at the interface level: same
    predict_* signatures regardless of the backend used.
    """

    def __init__(
        self,
        model_path: str,
        ref_columns_path: str,
        scaling_info_path: str,
        target_name_path: Optional[str] = None
    ):
        # --- Resolve and store paths (robust to relative/absolute inputs) ---
        self.model_path = _resolve(model_path)
        self.ref_columns_path = _resolve(ref_columns_path)
        self.scaling_info_path = _resolve(scaling_info_path)
        self.target_name_path = _resolve(target_name_path) if target_name_path else None

        # Validate that the model actually exists (fast-fail with a clear message).
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")

        # --- Load reference columns (feature order) ---
        with open(self.ref_columns_path, "r", encoding="utf-8") as f:
            self.ref_columns: List[str] = json.load(f)

        # --- Load scaling info (per-feature min/max for min–max normalization) ---
        with open(self.scaling_info_path, "r", encoding="utf-8") as f:
            self.scaling: Dict = json.load(f)

        # (Optional) load target name—useful for UI/consistency checks.
        self.target_name = None
        if self.target_name_path and os.path.isfile(self.target_name_path):
            with open(self.target_name_path, "r", encoding="utf-8") as f:
                self.target_name = f.read().strip()

        # Precompute min, max, and safe ranges as contiguous arrays (fast at inference).
        mins, maxs = [], []
        for c in self.ref_columns:
            s = self.scaling[c]
            mins.append(float(s["min"]))
            maxs.append(float(s["max"]))
        self._mins = np.array(mins, dtype=np.float32)
        self._maxs = np.array(maxs, dtype=np.float32)

        # Avoid division by zero: features with zero span get range=1 (acts like passthrough).
        self._range = np.where(self._maxs - self._mins == 0, 1.0, self._maxs - self._mins).astype(np.float32)

        # --- Load the model using the appropriate engine ---
        self.engine = self._pick_engine(self.model_path)
        if self.engine == "keras":
            if _keras_load is None:
                raise ImportError("TensorFlow/Keras not available to load a .keras/.h5 model.")
            self._keras = _keras_load(self.model_path)
            self._xgb = None
        else:
            if xgb is None:
                raise ImportError("xgboost is not installed but a .json/.ubj/.model was provided.")
            self._xgb = xgb.Booster()
            self._xgb.load_model(self.model_path)
            self._keras = None

    def _pick_engine(self, path: str) -> str:
        """
        Decide which backend to use based on file extension (or directory for SavedModel).
        Defaults to 'keras' if ambiguous to be conservative (most of my exports are Keras).
        """
        if os.path.isdir(path):
            return "keras"
        ext = os.path.splitext(path)[1].lower()
        if ext in (".keras", ".h5"):
            return "keras"
        if ext in (".json", ".ubj", ".model"):
            return "xgb"
        return "keras"

    # ----------------------------- Preprocessing -----------------------------

    def _scale_single(self, row: Dict[str, float]) -> np.ndarray:
        """
        Convert a single dict of feature->value into a (1, D) normalized array.

        Steps:
          1) Respect training feature order (`self.ref_columns`).
          2) Clip to [min, max] observed during training (reduces outlier risk).
          3) Apply min–max scaling to [0, 1].
        """
        raw = np.array([float(row[c]) for c in self.ref_columns], dtype=np.float32)
        clipped = np.clip(raw, self._mins, self._maxs)
        x = (clipped - self._mins) / self._range
        return x[None, :]  # shape: (1, D)

    def _scale_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Vectorized version of _scale_single for DataFrame inputs.

        Requirements:
          - df must contain at least the columns in self.ref_columns.
        """
        X = df[self.ref_columns].astype(np.float32).to_numpy()
        X = np.clip(X, self._mins, self._maxs)
        X = (X - self._mins) / self._range
        return X  # shape: (N, D)

    # ------------------------------- Inference -------------------------------

    def predict_single(self, row: Dict[str, float]) -> Tuple[float, float]:
        """
        Predict on a single example (dict), returning (p_positive, p_negative).

        I clamp the output into [0, 1] for safety, regardless of backend.
        For XGBoost I use DMatrix with explicit feature_names to lock column order.
        """
        X = self._scale_single(row)
        if self.engine == "keras":
            p_pos = float(self._keras.predict(X, verbose=0).reshape(-1)[0])
        else:
            dmat = xgb.DMatrix(X, feature_names=self.ref_columns)
            p_pos = float(self._xgb.predict(dmat).reshape(-1)[0])
        p_pos = min(max(p_pos, 0.0), 1.0)
        return p_pos, 1.0 - p_pos

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Batch predict on a DataFrame that contains the required feature set.

        Returns:
          A 1-D numpy array of positive-class probabilities (float32, clipped to [0, 1]).
        """
        X = self._scale_batch(df)
        if self.engine == "keras":
            p = self._keras.predict(X, verbose=0).reshape(-1).astype(np.float32)
        else:
            dmat = xgb.DMatrix(X, feature_names=self.ref_columns)
            p = self._xgb.predict(dmat).reshape(-1).astype(np.float32)
        return np.clip(p, 0.0, 1.0)

    # ------------------------------- Utilities -------------------------------

    @property
    def required_columns(self) -> List[str]:
        """
        Public accessor for the training-time feature order.
        Useful for validation and UI hints before inference.
        """
        return list(self.ref_columns)
