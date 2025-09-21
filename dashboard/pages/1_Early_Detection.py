# pages/1_Early_Detection.py
import json, os
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st
from joblib import load as joblib_load

from config import PATHS

# Streamlit requires this to be the first Streamlit command on the page
st.set_page_config(page_title="Early Detection", layout="wide")
st.title("Early Detection – Tabular Model")

"""
My Early Detection page:
- Loads a pre-trained RandomForest "risk" model (dietary/lifestyle proxy) plus its preprocessing artifacts.
- Supports CSV batch scoring and a single-case form (CSV can pre-fill the form).
- Ensures strict feature order and the exact training-time scaling/OHE used in my pipeline.
"""

# ----------------------- Load assets -----------------------
@st.cache_resource(show_spinner=False)
def load_assets():
    """
    Load the RF model and its preprocessing metadata from paths defined in config.py.

    I aggressively validate presence of:
      - ref_columns.json    (ordered list of features used at training time)
      - scaling_info.json   (numeric mean/scale, one-hot sources, mappings)
      - target_name.txt     (human-friendly target name; used for UI/consistency)
      - rf_model.joblib     (trained RandomForestClassifier)

    Returns:
      ref_columns, scaling_info, target_name, model
    """
    needed = {
        "ref_columns": PATHS.ED_REF_COLUMNS,
        "scaling_info": PATHS.ED_SCALING_INFO,
        "target_name": PATHS.ED_TARGET_NAME,
        "model_path": PATHS.ED_MODEL_PATH,
    }
    # Fail early if any path is missing or incorrect
    missing = [k for k, v in needed.items() if not v or not os.path.isfile(v)]
    if missing:
        raise FileNotFoundError(f"Missing files or wrong paths in config.py: {missing}")

    with open(PATHS.ED_REF_COLUMNS, "r", encoding="utf-8") as f:
        ref_columns: List[str] = json.load(f)
    with open(PATHS.ED_SCALING_INFO, "r", encoding="utf-8") as f:
        scaling_info: Dict = json.load(f)
    with open(PATHS.ED_TARGET_NAME, "r", encoding="utf-8") as f:
        target_name: str = f.read().strip()

    model = joblib_load(PATHS.ED_MODEL_PATH)
    return ref_columns, scaling_info, target_name, model

# Try to load once (cached). If anything fails, stop the page with a helpful message.
try:
    REF_COLS, SCALING, TARGET_NAME, RF_MODEL = load_assets()
except Exception as e:
    st.error("Could not load Early Detection assets. Check `config.py` paths.")
    st.exception(e)
    st.stop()

# ----------------------- Helpers -----------------------
# I store numeric scaling params and OHE specs from the training pipeline.
NUM_COLS: List[str] = SCALING["numeric"]["columns"]
MEAN = SCALING["numeric"]["mean"]
SCALE = SCALING["numeric"]["scale"]
OHE_SOURCES = SCALING["one_hot"]["source_categorical_columns"]
FAM_MAP = SCALING["mappings"]["Family_History_CRC"]  # {"Yes": 1, "No": 0}

def one_hot_groups_from_ref(ref_cols: List[str]) -> Dict[str, List[str]]:
    """
    Infer OHE group columns from training ref columns. I group by the prefix "<group>_".
    This lets me set one hot features robustly even if ordering changes.
    """
    groups: Dict[str, List[str]] = {g: [] for g in OHE_SOURCES}
    for c in ref_cols:
        for g in OHE_SOURCES:
            prefix = f"{g}_"
            if c.startswith(prefix):
                groups[g].append(c)
    return groups

OHE_GROUPS = one_hot_groups_from_ref(REF_COLS)

def normalize_cat(x: str) -> str:
    """Normalize free-text categorical inputs (trim + map common cases to canonical labels)."""
    if x is None:
        return ""
    x = str(x).strip()
    LUT = {"yes":"Yes", "no":"No", "male":"Male", "female":"Female"}
    return LUT.get(x.lower(), x.title())

def _set_one_hot(feat: Dict[str, float], group: str, choice: str):
    """
    Turn on exactly one OHE column per group if present in REF_COLS.
    If the expected column isn't present (drop_first), I leave zeros (baseline).
    """
    cols = OHE_GROUPS.get(group, [])
    wanted = f"{group}_{choice}"
    if wanted in cols:
        feat[wanted] = 1.0
    # else baseline (all zeros) is intentional

def build_feature_row(user_input: Dict[str, str | float]) -> pd.DataFrame:
    """
    Build a single-row feature DataFrame (1, F) matching REF_COLS order.

    Steps:
      1) Initialize all features to 0.
      2) Scale numeric features using (x - mean) / scale (guarding scale==0).
      3) Map Family_History_CRC to binary via FAM_MAP.
      4) Apply one-hot for all configured categorical sources.
    """
    feat = {c: 0.0 for c in REF_COLS}

    # Numeric (scaled)
    for col in NUM_COLS:
        raw = float(user_input[col])
        mu  = float(MEAN[col])
        sd  = float(SCALE[col])
        feat[col] = (raw - mu) / (sd if sd != 0 else 1.0)

    # Binary mapping (coerced to {"Yes","No"} with a fallback to "No")
    fam_val = normalize_cat(user_input["Family_History_CRC"])
    if fam_val not in ("Yes", "No"):
        fam_val = "No"
    feat["Family_History_CRC"] = float(FAM_MAP[str(fam_val)])

    # One-hot groups
    for group in OHE_SOURCES:
        choice = normalize_cat(user_input.get(group, ""))
        _set_one_hot(feat, group, choice)

    return pd.DataFrame([feat], columns=REF_COLS)

def predict_proba_pos(X: pd.DataFrame) -> np.ndarray:
    """
    Return P(positive) as a 1-D array regardless of how the model orders classes_.
    If classes_ doesn't include 1 explicitly (edge cases), I fall back safely.
    """
    proba = RF_MODEL.predict_proba(X)
    classes = list(getattr(RF_MODEL, "classes_", [0, 1]))
    if 1 in classes:
        pos_idx = classes.index(1)
    else:
        # If binary with weird order: use last column; else pick argmax per row.
        pos_idx = -1 if proba.shape[1] == 2 else int(np.argmax(proba, axis=1))
    return proba[:, pos_idx]

def coerce_choice(val: str, options: List[str], default: str) -> str:
    """Normalize and coerce a value into an allowed option set with a default fallback."""
    v = normalize_cat(val)
    return v if v in options else default

# Strictly required columns for CSV ingestion (and for prefill to work)
REQUIRED_INPUT_COLS = [
    "Age","BMI","Carbohydrates (g)","Proteins (g)","Fats (g)","Vitamin A (IU)",
    "Vitamin C (mg)","Iron (mg)","Family_History_CRC",
    "Gender","Lifestyle","Ethnicity","Pre-existing Conditions"
]

# ================================= CSV UPLOAD (TOP) ==========================
st.markdown("### Batch Prediction from CSV (also pre-fills the form below)")
st.caption("Required columns: " + ", ".join(REQUIRED_INPUT_COLS))

csv_file = st.file_uploader("Upload CSV", type=["csv"], key="ed_csv_uploader")

df_in = None
if csv_file is not None:
    # Read once; early-exit with a helpful error on failure
    try:
        df_in = pd.read_csv(csv_file)
    except Exception as e:
        st.error("Unable to read CSV.")
        st.exception(e)
        st.stop()

    # Check the schema strictly so I can guarantee mapping and scaling logic
    missing = [c for c in REQUIRED_INPUT_COLS if c not in df_in.columns]
    if missing:
        st.error("CSV is missing required columns: " + ", ".join(missing))
        st.stop()

    # Row selector to pre-fill the single prediction form below
    if len(df_in) > 0:
        # Remember the selected row between reruns
        if "ed_row_index" not in st.session_state:
            st.session_state["ed_row_index"] = 0
        st.session_state["ed_row_index"] = st.number_input(
            "Select row to pre-fill the form below",
            min_value=0,
            max_value=len(df_in)-1,
            value=st.session_state["ed_row_index"],
            step=1,
            help="Changing this updates the Single Prediction form fields."
        )

        # Prefill: store values in session_state BEFORE widgets render
        r = df_in.iloc[st.session_state["ed_row_index"]]
        # numeric
        st.session_state["Age"]  = int(r["Age"])
        st.session_state["BMI"]  = float(r["BMI"])
        st.session_state["Carbohydrates (g)"] = float(r["Carbohydrates (g)"])
        st.session_state["Proteins (g)"]      = float(r["Proteins (g)"])
        st.session_state["Fats (g)"]          = float(r["Fats (g)"])
        st.session_state["Vitamin A (IU)"]    = float(r["Vitamin A (IU)"])
        st.session_state["Vitamin C (mg)"]    = float(r["Vitamin C (mg)"])
        st.session_state["Iron (mg)"]         = float(r["Iron (mg)"])
        # categoricals with coercion
        st.session_state["Family_History_CRC"] = coerce_choice(r["Family_History_CRC"], ["No","Yes"], "No")
        st.session_state["Gender"]             = coerce_choice(r["Gender"], ["Male","Female"], "Male")
        st.session_state["Lifestyle"]          = coerce_choice(r["Lifestyle"], ["Moderate Exercise","Sedentary","Smoker"], "Moderate Exercise")
        st.session_state["Ethnicity"]          = coerce_choice(r["Ethnicity"], ["Asian","Caucasian","Hispanic"], "Caucasian")
        st.session_state["Pre-existing Conditions"] = coerce_choice(r["Pre-existing Conditions"], ["None","Hypertension","Obesity"], "None")

# ================================= SINGLE PREDICTION =========================
st.markdown("---")
st.markdown("### Single Prediction")

# Widgets bind to session_state keys so CSV prefill shows up automatically
c1, c2, c3 = st.columns(3)
with c1:
    st.number_input("Age", min_value=1, max_value=120, step=1, key="Age")
    st.number_input("BMI", min_value=10.0, max_value=60.0, step=0.1, format="%.1f", key="BMI")
    st.number_input("Carbohydrates (g)", min_value=0.0, max_value=1000.0, step=1.0, key="Carbohydrates (g)")
with c2:
    st.number_input("Proteins (g)", min_value=0.0, max_value=400.0, step=0.5, key="Proteins (g)")
    st.number_input("Fats (g)", min_value=0.0, max_value=400.0, step=0.5, key="Fats (g)")
    st.number_input("Vitamin A (IU)", min_value=0.0, max_value=50000.0, step=10.0, key="Vitamin A (IU)")
with c3:
    st.number_input("Vitamin C (mg)", min_value=0.0, max_value=2000.0, step=1.0, key="Vitamin C (mg)")
    st.number_input("Iron (mg)", min_value=0.0, max_value=100.0, step=0.1, key="Iron (mg)")
    st.selectbox("Family History of CRC", ["No", "Yes"], key="Family_History_CRC")

# Categorical: fixed option sets (normalized upstream if CSV provided)
st.selectbox("Gender", ["Male", "Female"], key="Gender")
st.selectbox("Lifestyle", ["Moderate Exercise", "Sedentary", "Smoker"], key="Lifestyle")
st.selectbox("Ethnicity", ["Asian", "Caucasian", "Hispanic"], key="Ethnicity")
st.selectbox("Pre-existing Conditions", ["None", "Hypertension", "Obesity"], key="Pre-existing Conditions")

if st.button("Predict", type="primary"):
    # Collect current form values
    ui = {
        "Age": st.session_state["Age"],
        "BMI": st.session_state["BMI"],
        "Carbohydrates (g)": st.session_state["Carbohydrates (g)"],
        "Proteins (g)": st.session_state["Proteins (g)"],
        "Fats (g)": st.session_state["Fats (g)"],
        "Vitamin A (IU)": st.session_state["Vitamin A (IU)"],
        "Vitamin C (mg)": st.session_state["Vitamin C (mg)"],
        "Iron (mg)": st.session_state["Iron (mg)"],
        "Family_History_CRC": st.session_state["Family_History_CRC"],
        "Gender": st.session_state["Gender"],
        "Lifestyle": st.session_state["Lifestyle"],
        "Ethnicity": st.session_state["Ethnicity"],
        "Pre-existing Conditions": st.session_state["Pre-existing Conditions"],
    }
    try:
        X = build_feature_row(ui)
        p_mal = float(predict_proba_pos(X)[0])  # positive class probability
    except Exception as e:
        st.error("Prediction failed.")
        st.exception(e)
        st.stop()

    p_ben = 1.0 - p_mal
    pred_label = "at risk (positive)" if p_mal >= 0.5 else "not at risk (negative)"

    st.subheader("Result")
    # Quick headline metric; I append the numeric probability as delta for visibility.
    st.metric("Early Detection (Tabular)", pred_label, delta=f"prob: {p_mal:.3f}")

    b1, b2 = st.columns(2)
    with b1:
        st.write(f"**At-risk probability:** {p_mal:.1%}")
        st.progress(min(max(p_mal, 0.0), 1.0))
    with b2:
        st.write(f"**Not at risk probability:** {p_ben:.1%}")
        st.progress(min(max(p_ben, 0.0), 1.0))

# ================================= BATCH PREDICTION ==========================
if csv_file is not None:
    st.markdown("---")
    st.markdown("### Batch Prediction Results (from uploaded CSV)")

    # Normalize categorical text and build features row-by-row (preserves strict order)
    rows = []
    for _, r in df_in.iterrows():
        ui = {
            "Age": r["Age"],
            "BMI": r["BMI"],
            "Carbohydrates (g)": r["Carbohydrates (g)"],
            "Proteins (g)": r["Proteins (g)"],
            "Fats (g)": r["Fats (g)"],
            "Vitamin A (IU)": r["Vitamin A (IU)"],
            "Vitamin C (mg)": r["Vitamin C (mg)"],
            "Iron (mg)": r["Iron (mg)"],
            "Family_History_CRC": normalize_cat(r["Family_History_CRC"]),
            "Gender": normalize_cat(r["Gender"]),
            "Lifestyle": normalize_cat(r["Lifestyle"]),
            "Ethnicity": normalize_cat(r["Ethnicity"]),
            "Pre-existing Conditions": normalize_cat(r["Pre-existing Conditions"]),
        }
        X_row = build_feature_row(ui)
        rows.append(X_row)

    X_all = pd.concat(rows, axis=0).reset_index(drop=True)

    # Predict all rows at once
    try:
        p_pos = predict_proba_pos(X_all)
    except Exception as e:
        st.error("Batch prediction failed.")
        st.exception(e)
        st.stop()

    pred_label = np.where(p_pos >= 0.5, "at risk (positive)", "not at risk (negative)")
    out = df_in.copy()
    out["prob_positive"] = p_pos
    out["prob_negative"] = 1.0 - p_pos
    out["prediction"] = pred_label

    st.success(f"Predicted {len(out)} rows.")
    st.dataframe(out.head(50), use_container_width=True)

    # Allow users to download the full results
    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download predictions CSV",
        data=csv_bytes,
        file_name="early_detection_predictions.csv",
        mime="text/csv",
    )

st.caption(
    "Uses training column order from `ref_columns.json`, scaling from `scaling_info.json`, "
    "and binary mapping for Family_History_CRC; model loaded from `.joblib`."
)
