# coimbra_model.py
import json, os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# ----------------------- Coimbra Tabular Model -----------------------
class CoimbraTabularModel:
    """
    CoimbraTabularModel
    -------------------
    My unified, minimal wrapper around a Keras binary classifier trained on the
    Coimbra dataset (tabular). It loads:
      - The model (.keras)
      - Reference feature columns (JSON list; order must match training)
      - Min–max scaling info (per-feature min/max from training)
      - Optional target name (text file; used for UI/sanity checks)

    I expose helpers for single and batch predictions with exactly the same
    preprocessing I used during training (clip to [min, max] then min–max scale).
    """

    def __init__(
        self,
        model_path: str,
        ref_columns_path: str,
        scaling_info_path: str,
        target_name_path: str | None = None,
    ):
        # Fast-fail on missing inputs so misconfigurations are obvious.
        if not (os.path.isfile(model_path) and os.path.isfile(ref_columns_path) and os.path.isfile(scaling_info_path)):
            raise FileNotFoundError("Coimbra model or preprocessing files not found.")

        # ---- Load preprocessing metadata ----
        # Reference column order (this *must* match training-time order).
        with open(ref_columns_path, "r", encoding="utf-8") as f:
            self.ref_columns: List[str] = json.load(f)

        # Per-feature scaling info: {"feature": {"min": ..., "max": ...}, ...}
        with open(scaling_info_path, "r", encoding="utf-8") as f:
            self.scaling: Dict = json.load(f)

        # Optional: human-readable target name (useful for UI labeling).
        self.target_name = None
        if target_name_path and os.path.isfile(target_name_path):
            with open(target_name_path, "r", encoding="utf-8") as f:
                self.target_name = f.read().strip()

        # ---- Load the trained Keras model ----
        # Assumes last layer is a single sigmoid unit (binary probability).
        self.model = load_model(model_path)

        # ---- Cache arrays for vectorized transforms ----
        # I precompute mins, maxs, and safe ranges to keep inference fast and robust.
        self._mins = np.array(
            [float(self.scaling[c]["min"]) for c in self.ref_columns], dtype=np.float32
        )
        self._maxs = np.array(
            [float(self.scaling[c]["max"]) for c in self.ref_columns], dtype=np.float32
        )
        # Guard against zero-range features to avoid divide-by-zero (range=1 acts like passthrough).
        self._ranges = np.where(self._maxs - self._mins == 0, 1.0, self._maxs - self._mins).astype(np.float32)

    # ------------------ Preprocess ------------------
    def _to_feature_array(self, row_dict: Dict[str, float | int]) -> np.ndarray:
        """
        Convert a single example (dict) into a (1, F) float32 array using my training order
        and min–max scaling:

            x_scaled = (clip(x, min, max) - min) / (max - min)

        Any value outside [min, max] is clipped to reduce sensitivity to outliers.
        """
        raw = np.array([float(row_dict[c]) for c in self.ref_columns], dtype=np.float32)
        clipped = np.clip(raw, self._mins, self._maxs)
        x = (clipped - self._mins) / self._ranges
        return x[None, :]  # shape = (1, F)

    # ------------------ Predict ------------------
    def predict_single(self, row_dict: Dict[str, float | int]) -> Tuple[float, float]:
        """
        Predict on a single example and return (p_positive, p_negative).

        I assume a single sigmoid output in the model, so p_positive is in [0, 1].
        I explicitly clamp it for safety.
        """
        x = self._to_feature_array(row_dict)
        p_pos = float(self.model.predict(x, verbose=0).reshape(-1)[0])
        p_pos = max(0.0, min(1.0, p_pos))
        return p_pos, 1.0 - p_pos

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        """
        Batch prediction on a DataFrame.

        Requirements:
          - df must contain at least the columns listed in self.ref_columns.
          - Extra columns are ignored; column order is enforced by selection.

        Returns:
          - A 1-D float32 numpy array of positive-class probabilities in [0, 1].
        """
        X = df[self.ref_columns].astype(np.float32).to_numpy()
        X = np.clip(X, self._mins, self._maxs)
        X = (X - self._mins) / self._ranges
        p_pos = self.model.predict(X, verbose=0).reshape(-1).astype(np.float32)
        p_pos = np.clip(p_pos, 0.0, 1.0)
        return p_pos

    # ------------------ Convenience ------------------
    @property
    def required_columns(self) -> List[str]:
        """
        The exact training-time feature order expected by the model.
        I expose this so callers can validate inputs before inference.
        """
        return list(self.ref_columns)
