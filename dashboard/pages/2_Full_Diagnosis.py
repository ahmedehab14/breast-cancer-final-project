# pages/2_Full_Diagnosis.py
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

import numpy as np
from PIL import Image
import pandas as pd
import streamlit as st
from io import BytesIO
from datetime import datetime

from config import PATHS, CFG
from cbis_ddsm_mammography import CBISDDSMPipeline
from coimbra_model import CoimbraTabularModel
from wisconsin_model import WisconsinTabularModel  # unified (.keras or XGBoost .json) loader

# Try to import reportlab for PDF export (optional dependency)
try:
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.utils import ImageReader
    from reportlab.lib import colors
    _PDF_OK = True
except Exception:
    _PDF_OK = False

# Streamlit requires this to be the first Streamlit command on the page
st.set_page_config(page_title="Full Diagnosis — Staged Workflow", layout="wide")
st.title("Full Diagnosis — Staged Workflow")
st.caption("For research/education use only. Not a medical device; not for clinical decisions.")

"""
My full diagnosis page implements a staged workflow:

• Stage 1 (Imaging): YOLO → ROI crop → MaxViT-Tiny(+metadata) classification → explanations (Grad-CAM++ & gradient SHAP)
• Stage 2 (Tabular): Coimbra Keras model (min–max scaling, strict feature order)
• Stage 3 (Tabular): Wisconsin (Keras or XGBoost; unified loader)
• Fusion: Late-fusion of available stage probabilities with configurable method + weights
• Export: Optional PDF report via reportlab (if available)

Design choices:
- I keep each stage self-contained and persist results in session_state (S1/S2/S3) for later fusion/export.
- A ModelRegistry separates image and tabular specs, so swapping models/paths is straightforward.
- Defensive guards: I validate presence of model files early and fail with clear messages.
"""

# --------------------------- utils & session ---------------------------
def _badge(ok: bool) -> str:
    """Small helper: pretty status label for stage completion."""
    return "✅ Completed" if ok else "⏳ Pending"

def _fmt_pct(x: float) -> str:
    """Format a float probability as a percentage string with one decimal place."""
    return f"{100.0*float(x):.1f}%"

# Stage result slots in session_state (I initialize only if missing)
if "S1" not in st.session_state:  # Stage 1 (Imaging)
    st.session_state["S1"] = None
if "S2" not in st.session_state:  # Stage 2 (Coimbra)
    st.session_state["S2"] = None
if "S3" not in st.session_state:  # Stage 3 (Wisconsin)
    st.session_state["S3"] = None

# --------------------------- registries ---------------------------
@dataclass
class ImageModelSpec:
    """Spec for Stage 1 imaging: YOLO weights and classifier weights."""
    name: str
    yolo_path: str
    clf_path: str

@dataclass
class TabularModelSpec:
    """Spec for tabular stages: loader type + model and preprocessing artifacts."""
    name: str
    loader: str  # "coimbra" | "wisconsin"
    model_path: str
    ref_columns: str
    scaling_info: str
    target_name: str

class ModelRegistry:
    """
    Thin registry abstraction that lets me register/retrieve image and tabular models
    by a user-facing key. Keeps the sidebar UX simple and the core code clean.
    """
    def __init__(self):
        self._image: Dict[str, ImageModelSpec] = {}
        self._tabular: Dict[str, TabularModelSpec] = {}

    def register_image(self, key: str, spec: ImageModelSpec): self._image[key] = spec
    def register_tabular(self, key: str, spec: TabularModelSpec): self._tabular[key] = spec
    def get_type(self, key: str) -> str:
        if key in self._image: return "image"
        if key in self._tabular: return "tabular"
        return "unknown"
    def img(self, key: str) -> Optional[ImageModelSpec]: return self._image.get(key)
    def tab(self, key: str) -> Optional[TabularModelSpec]: return self._tabular.get(key)

# Declare my registry and add available models using central config paths
REG = ModelRegistry()
# Imaging (Stage 1)
REG.register_image("Calcification (Image)", ImageModelSpec("Calcification", PATHS.CALC_YOLO, PATHS.CALC_MAXV))
REG.register_image("Mass (Image)",          ImageModelSpec("Mass",          PATHS.MASS_YOLO, PATHS.MASS_MAXV))
# Tabular (Stage 2/3)
REG.register_tabular("Coimbra (Tabular)", TabularModelSpec(
    name="Coimbra", loader="coimbra",
    model_path=PATHS.COIMBRA_MODEL, ref_columns=PATHS.COIMBRA_REF_COLUMNS,
    scaling_info=PATHS.COIMBRA_SCALING_INFO, target_name=PATHS.COIMBRA_TARGET_NAME
))
REG.register_tabular("Wisconsin (Tabular)", TabularModelSpec(
    name="Wisconsin", loader="wisconsin",
    model_path=PATHS.WISC_MODEL, ref_columns=PATHS.WISC_REF_COLUMNS,
    scaling_info=PATHS.WISC_SCALING_INFO, target_name=PATHS.WISC_TARGET_NAME
))

# --------------------------- sidebar ---------------------------
with st.sidebar:
    st.header("Workflow")
    # Stage selector controls which section renders
    stage = st.selectbox(
        "Select stage",
        [
            "Stage 1 — Mammogram Scanning",
            "Stage 2 — Coimbra (Tabular)",
            "Stage 3 — Wisconsin (Tabular)",
            "Overall Assessment (Fusion)"
        ],
        index=0,
    )
    imaging_choice = None
    if stage.startswith("Stage 1"):
        # Imaging sub-choice (calcification vs mass)
        imaging_choice = st.selectbox("Imaging modality", ["Calcification (Image)", "Mass (Image)"], index=0)

# Map sidebar selection to internal mode/model key
if stage.startswith("Stage 1"):
    model_choice = imaging_choice or "Calcification (Image)"
    mode = "image"
elif stage.startswith("Stage 2"):
    model_choice = "Coimbra (Tabular)"
    mode = "tabular"
elif stage.startswith("Stage 3"):
    model_choice = "Wisconsin (Tabular)"
    mode = "tabular"
else:
    model_choice = "FUSION"
    mode = "fusion"

# =============================== IMAGE FLOW (S1) ==============================
if mode == "image":
    spec = REG.img(model_choice)
    if not spec:
        st.error("Imaging model not found."); st.stop()

    # Validate that both YOLO and classifier weights exist before constructing pipeline
    for pth, name in [(spec.yolo_path, "YOLO detector"), (spec.clf_path, "classification network")]:
        if not pth or not os.path.isfile(pth):
            st.warning(f"{name} weights not found: {pth or '(empty)'} — update paths in config.py.")
            st.stop()

    @st.cache_resource(show_spinner=False)
    def get_pipeline(yolo_path: str, clf_path: str, thresh: float) -> CBISDDSMPipeline:
        """Cache the imaging pipeline to avoid reloading weights on every rerun."""
        return CBISDDSMPipeline(yolo_path, clf_path, thresh=thresh)

    # Imaging metadata inputs (used by my classifier as a 4-D one-hot)
    c1, c2 = st.columns(2)
    with c1:
        laterality = st.radio("Breast laterality", ["LEFT", "RIGHT"], horizontal=True, index=0)
    with c2:
        view = st.radio("Projection", ["CC", "MLO"], horizontal=True, index=0)

    # Build pipeline (YOLO + MaxViT) with my operating threshold from CFG
    try:
        pipe = get_pipeline(spec.yolo_path, spec.clf_path, CFG.THRESH)
    except Exception as e:
        st.error("Model initialisation failed."); st.exception(e); st.stop()

    # Classifier expects metadata set before classify()/explain()
    pipe.set_metadata(laterality, view)

    # Full-field mammogram upload (I accept common image formats)
    uploaded = st.file_uploader(
        "Upload full-field digital mammogram",
        type=["png", "jpg", "jpeg", "bmp", "tif", "tiff"],
        key="diag_img_uploader",
    )
    if not uploaded:
        st.info("Please provide a mammogram to proceed."); st.stop()

    pil_full = Image.open(uploaded).convert("RGB")
    full_rgb = np.array(pil_full)

    # --- Stage 1: detection ---
    try:
        with st.spinner("Localising region of interest (YOLO)…"):
            annot_rgb, crop_rgb, bbox = pipe.detect(full_rgb, conf=CFG.CONF_THR, iou=CFG.IOU_THR)
    except Exception as e:
        st.error("Detection step failed."); st.exception(e); st.stop()

    # Side-by-side: raw input and YOLO overlay
    colL, colR = st.columns(2); display_w = 420
    with colL:
        st.subheader("Input image"); st.image(pil_full, width=display_w)
    with colR:
        st.subheader(f"Detected ROI — {spec.name}")
        if annot_rgb is not None: st.image(annot_rgb, width=display_w)
        else: st.warning("No candidate lesion detected.")

    if crop_rgb is None:
        # Respectful exit when detector finds nothing
        st.warning(f"No {spec.name.lower()} lesion identified by the detector."); st.stop()

    st.subheader("Region of interest (cropped)"); st.image(crop_rgb, width=display_w)

    # --- Stage 1: classification ---
    try:
        with st.spinner("Characterising lesion…"):
            label, conf, p_mal, x_tensor = pipe.classify(crop_rgb)
    except Exception as e:
        st.error("Classification step failed."); st.exception(e); st.stop()

    st.subheader("Interpretation")
    st.metric(label=f"Imaging classifier — {spec.name}", value=label.capitalize(), delta=f"confidence: {conf:.3f}")

    p_ben = 1.0 - float(p_mal)
    b1, b2 = st.columns(2)
    with b1:
        st.write(f"**Probability of malignancy:** {_fmt_pct(p_mal)}")
        st.progress(min(max(float(p_mal), 0.0), 1.0))
    with b2:
        st.write(f"**Probability of benign pathology:** {_fmt_pct(p_ben)}")
        st.progress(min(max(p_ben, 0.0), 1.0))

    # --- Stage 1: explanations (Grad-CAM++ & gradient SHAP) ---
    try:
        with st.spinner("Generating visual explanations…"):
            cam_img, shap_img = pipe.explain(x_tensor)
    except Exception as e:
        st.error("Explanation step failed."); st.exception(e); st.stop()

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Grad-CAM++ (predicted class)"); st.image(cam_img, width=display_w)
    with c4:
        st.subheader("GradientSHAP"); st.image(shap_img, width=display_w)

    # ---- Persist Stage 1 results for fusion/export ----
    st.session_state["S1"] = {
        "modality": spec.name,
        "laterality": laterality,
        "view": view,
        "p_malignant": float(p_mal),
        "label": label,
        "confidence": float(conf),
        "bbox": bbox,
        "original_full": full_rgb,       # save original (RGB)
        "annot_full": annot_rgb,         # YOLO overlay (RGB)
        "crop": crop_rgb,                # ROI (RGB)
        "gradcam": cam_img,              # (RGB)
        "shap": shap_img,                # (RGB)
        "yolo_weights": os.path.basename(spec.yolo_path),
        "clf_weights": os.path.basename(spec.clf_path),
    }
    st.success("Stage 1 saved to session.")

# =============================== TABULAR FLOW (S2/S3) =========================
elif mode == "tabular":
    spec = REG.tab(model_choice)
    if not spec:
        st.error("Tabular model not found."); st.stop()

    # Validate required files (model + preprocessing artifacts)
    needed = [
        (spec.model_path, f"{spec.name} model"),
        (spec.ref_columns, "ref_columns.json"),
        (spec.scaling_info, "scaling_info.json"),
    ]
    missing = [name for p, name in needed if not p or not os.path.isfile(p)]
    if missing:
        st.warning("Missing files: " + ", ".join(missing) + ". Update paths in config.py.")
        st.stop()

    @st.cache_resource(show_spinner=False)
    def get_tabular_model(loader, mpath, rpath, spath, tpath):
        """
        Cache the appropriate tabular model wrapper based on `loader`.
        Coimbra → Keras; Wisconsin → unified (Keras or XGBoost).
        """
        if loader == "coimbra":
            return CoimbraTabularModel(mpath, rpath, spath, tpath)
        elif loader == "wisconsin":
            return WisconsinTabularModel(mpath, rpath, spath, tpath)
        else:
            raise ValueError(f"Unknown tabular loader: {loader}")

    # Construct once (cached) and surface any init errors cleanly
    try:
        tmodel = get_tabular_model(spec.loader, spec.model_path, spec.ref_columns, spec.scaling_info, spec.target_name)
    except Exception as e:
        st.error(f"Failed to initialise {spec.name} model."); st.exception(e); st.stop()

    st.subheader(f"{spec.name} — Single-case and Batch Evaluation")

    # User guidance for expected CSV schema per dataset
    if spec.name == "Coimbra":
        st.markdown("**Upload CSV (optional) — must contain:** Age, BMI, Glucose, Insulin, HOMA, Leptin, Adiponectin, Resistin, MCP.1")
    elif spec.name == "Wisconsin":
        st.markdown(
            "**Upload CSV (optional) — must contain:** "
            "radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, "
            "compactness_mean, concavity_mean, concave points_mean, symmetry_mean, "
            "fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, "
            "smoothness_se, compactness_se, concavity_se, concave points_se, symmetry_se, "
            "fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, "
            "smoothness_worst, compactness_worst, concavity_worst, concave points_worst, "
            "symmetry_worst, fractal_dimension_worst"
        )

    # CSV upload (optional) and prefill of numeric inputs
    csv_file = st.file_uploader(f"Upload {spec.name} CSV", type=["csv"], key=f"{spec.name}_csv_uploader")
    df_in = None
    if csv_file is not None:
        try:
            df_in = pd.read_csv(csv_file)
        except Exception as e:
            st.error("CSV could not be read."); st.exception(e); st.stop()
        # Hard schema check against training-time required columns
        missing_cols = [c for c in tmodel.required_columns if c not in df_in.columns]
        if missing_cols:
            st.error("CSV is missing required columns: " + ", ".join(missing_cols)); st.stop()
        # Optionally prefill the single-case form from a selected row
        if len(df_in) > 0:
            row_key = f"{spec.name}_row"
            if row_key not in st.session_state: st.session_state[row_key] = 0
            st.session_state[row_key] = st.number_input(
                "Select record to pre-fill fields", min_value=0, max_value=len(df_in)-1,
                value=st.session_state[row_key], step=1,
            )
            r = df_in.iloc[st.session_state[row_key]]
            for col in tmodel.required_columns:
                st.session_state[f"TAB::{spec.name}::{col}"] = float(r[col])

    # Dynamic numeric inputs rendered in a 3-column grid (order = training order)
    cols = tmodel.required_columns
    grid = st.columns(3)
    for i, col in enumerate(cols):
        with grid[i % 3]:
            st.number_input(col, min_value=0.0, step=0.001, key=f"TAB::{spec.name}::{col}")

    # Single-case evaluation
    if st.button(f"Run {spec.name} evaluation", type="primary"):
        row = {c: float(st.session_state.get(f"TAB::{spec.name}::{c}", 0.0)) for c in cols}
        try:
            p_pos, p_neg = tmodel.predict_single(row)
        except Exception as e:
            st.error("Prediction failed."); st.exception(e); st.stop()

        label = "Malignancy likely" if p_pos >= 0.5 else "Malignancy unlikely"
        st.subheader("Interpretation")
        st.metric(f"{spec.name} model", label, delta=f"probability of malignancy: {p_pos:.3f}")

        b1, b2 = st.columns(2)
        with b1:
            st.write(f"**Probability of malignancy:** {_fmt_pct(p_pos)}"); st.progress(min(max(p_pos, 0.0), 1.0))
        with b2:
            st.write(f"**Probability of non-malignant outcome:** {_fmt_pct(p_neg)}"); st.progress(min(max(p_neg, 0.0), 1.0))

        # ---- Persist Stage 2/3 results for fusion/export ----
        store_key = "S2" if spec.name == "Coimbra" else "S3"
        st.session_state[store_key] = {
            "model": spec.name,
            "features": row,                 # numeric dict used
            "p_malignant": float(p_pos),
        }
        st.success(f"{spec.name}: results saved to session.")

    # Batch evaluation mirrors the single-case path using tmodel.predict_batch
    if df_in is not None:
        st.markdown("---")
        st.markdown(f"#### Batch evaluation — {spec.name} CSV")
        try:
            p_pos = tmodel.predict_batch(df_in)
        except Exception as e:
            st.error("Batch prediction failed."); st.exception(e); st.stop()

        out = df_in.copy()
        out["probability_malignant"] = p_pos
        out["probability_non_malignant"] = 1.0 - p_pos
        out["assessment"] = np.where(p_pos >= 0.5, "Malignancy likely", "Malignancy unlikely")
        st.success(f"Evaluated {len(out)} record(s).")
        st.dataframe(out.head(50), use_container_width=True)
        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button(
            f"Download {spec.name} batch results", data=csv_bytes,
            file_name=f"{spec.name.lower()}_predictions.csv", mime="text/csv",
        )

# =============================== FUSION (Overall) =============================
elif mode == "fusion":
    st.subheader("Overall Assessment (Fusion)")
    s1 = st.session_state.get("S1")
    s2 = st.session_state.get("S2")
    s3 = st.session_state.get("S3")

    # Quick view: which stages are completed
    colA, colB, colC = st.columns(3)
    with colA: st.write("**Stage 1 — Mammogram**:", _badge(s1 is not None))
    with colB: st.write("**Stage 2 — Coimbra**:", _badge(s2 is not None))
    with colC: st.write("**Stage 3 — Wisconsin**:", _badge(s3 is not None))

    # Collect available probabilities (I fuse only what exists)
    parts = []
    if s1 is not None:
        parts.append(("Imaging", float(s1["p_malignant"])))
    if s2 is not None:
        parts.append(("Coimbra", float(s2["p_malignant"])))
    if s3 is not None:
        parts.append(("Wisconsin", float(s3["p_malignant"])))

    if not parts:
        st.info("No stage results available. Please complete one or more stages first."); st.stop()

    # Fusion method and per-component weights
    st.markdown("#### Fusion strategy")
    strategy = st.selectbox("Method", ["Average (late fusion)", "Noisy-OR", "Max rule"], index=0)

    st.markdown("#### Weights")
    cW1, cW2, cW3 = st.columns(3)
    default_w = {"Imaging": 0.5, "Coimbra": 0.25, "Wisconsin": 0.25}
    weights = {}
    for name, _ in parts:
        with (cW1 if name == "Imaging" else cW2 if name == "Coimbra" else cW3):
            weights[name] = st.slider(f"{name} weight", 0.0, 1.0, float(default_w.get(name, 0.33)), 0.01)
    # Normalize to sum to 1 (avoid user errors)
    s = sum(weights.values()) or 1.0
    weights = {k: v / s for k, v in weights.items()}

    # Compute fused probability using the selected strategy
    p_map = dict(parts)
    if strategy.startswith("Average"):
        p_fused = sum(weights[n]*p_map[n] for n in weights)
    elif strategy.startswith("Noisy"):
        term = 1.0
        for n in weights:
            term *= (1.0 - weights[n]*p_map[n])
        p_fused = 1.0 - term
    else:  # Max rule
        p_fused = max(p_map[n] for n in weights)

    st.markdown("#### Overall result")
    thresh = st.slider("Decision threshold", 0.0, 1.0, 0.5, 0.01)
    label = "Malignancy likely" if p_fused >= thresh else "Malignancy unlikely"

    cF1, cF2 = st.columns(2)
    with cF1:
        st.metric("Fused probability of malignancy", _fmt_pct(p_fused), delta=f"threshold: {thresh:.2f}")
    with cF2:
        st.metric("Overall assessment", label)

    # Component visual bars (one per available stage)
    st.markdown("#### Component probabilities")
    cc1, cc2, cc3 = st.columns(3)
    for (name, p), col in zip(parts, (cc1, cc2, cc3)):
        with col:
            st.write(f"**{name}:** {_fmt_pct(p)}")
            st.progress(min(max(p, 0.0), 1.0))

    # --------------- PDF generation ---------------
    st.markdown("---")
    st.markdown("#### Generate overall PDF report")

    if not _PDF_OK:
        st.error("PDF export requires `reportlab`. Install with: `pip install reportlab`")
    else:
        if st.button("Generate PDF report (timestamped)", type="primary"):
            # Build the PDF entirely in memory (no disk I/O)
            pdf_bytes = BytesIO()
            c = canvas.Canvas(pdf_bytes, pagesize=A4)
            PW, PH = A4
            M = 36  # 0.5 inch margin
            TS = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            # ---- Small PDF helpers ----
            def draw_header(title: str):
                """Standard header with title + generation timestamp and a divider line."""
                c.setFont("Helvetica-Bold", 14)
                c.drawString(M, PH - M, title)
                c.setFont("Helvetica", 10)
                c.drawRightString(PW - M, PH - M, f"Generated: {TS}")
                c.setStrokeColor(colors.grey)
                c.line(M, PH - M - 6, PW - M, PH - M - 6)

            def draw_kv(lines: List[str], x, y, leading=12, font="Helvetica", size=10):
                """Draw simple line-by-line key/value text; returns the new y cursor."""
                c.setFont(font, size)
                yy = y
                for ln in lines:
                    c.drawString(x, yy, ln)
                    yy -= leading
                return yy

            def to_img_reader(arr_or_pil):
                """
                Convert numpy (RGB uint8) or PIL.Image to a reportlab ImageReader.
                Returns None if input is None or unsupported.
                """
                if arr_or_pil is None:
                    return None
                if isinstance(arr_or_pil, np.ndarray):
                    pil = Image.fromarray(arr_or_pil.astype(np.uint8))
                elif isinstance(arr_or_pil, Image.Image):
                    pil = arr_or_pil
                else:
                    return None
                return ImageReader(pil)

            def draw_image(img_reader, x, y, max_w, max_h):
                """Draw an image preserving aspect ratio within a bounding box."""
                if img_reader is None:
                    return 0, 0
                iw, ih = img_reader.getSize()
                scale = min(max_w/iw, max_h/ih)
                w, h = iw*scale, ih*scale
                c.drawImage(img_reader, x, y, width=w, height=h, preserveAspectRatio=True, mask='auto')
                return w, h

            # ----- Cover / Summary -----
            draw_header("Overall Assessment Report")
            y = PH - M - 24
            c.setFont("Helvetica", 11)
            c.drawString(M, y, "For research/education only. Not a medical device.")
            y -= 18
            c.drawString(M, y, f"Fusion: {strategy}  |  Threshold: {thresh:.2f}  |  Fused Malignancy Probability: {_fmt_pct(p_fused)}")
            y -= 14
            wtxt = ", ".join([f"{k}={weights[k]:.2f}" for k in weights])
            c.drawString(M, y, f"Weights: {wtxt}")
            y -= 28
            c.setFont("Helvetica-Bold", 12)
            c.drawString(M, y, "Stage completion")
            y -= 16
            c.setFont("Helvetica", 11)
            c.drawString(M, y, f"Stage 1 — Mammogram: {'Completed' if s1 else 'Pending'}")
            y -= 14
            c.drawString(M, y, f"Stage 2 — Coimbra:   {'Completed' if s2 else 'Pending'}")
            y -= 14
            c.drawString(M, y, f"Stage 3 — Wisconsin: {'Completed' if s3 else 'Pending'}")
            c.showPage()

            # ----- Stage 1 page(s) -----
            if s1:
                draw_header("Stage 1 — Mammogram Scanning (Imaging)")
                y = PH - M - 28
                lines = [
                    f"Modality: {s1['modality']}   |   Laterality: {s1['laterality']}   |   Projection: {s1['view']}",
                    f"Classifier label: {s1['label'].capitalize()}   |   Confidence: {s1['confidence']:.3f}   |   P(malignant): {_fmt_pct(s1['p_malignant'])}",
                    f"YOLO weights: {s1['yolo_weights']}   |   Classifier: {s1['clf_weights']}",
                ]
                y = draw_kv(lines, M, y, leading=14) - 8

                col_w = (PW - 2*M - 12) / 2.0
                row_h = 250

                # Row 1: Original vs YOLO ROI
                img1 = to_img_reader(s1.get("original_full"))
                img2 = to_img_reader(s1.get("annot_full"))
                draw_image(img1, M, y - row_h, col_w, row_h)
                draw_image(img2, M + col_w + 12, y - row_h, col_w, row_h)
                c.setFont("Helvetica", 9)
                c.drawString(M, y - row_h - 12, "Original image")
                c.drawString(M + col_w + 12, y - row_h - 12, "YOLO: full image with ROI overlay")
                c.showPage()

                # Row 2 & 3: Crop, Grad-CAM++, SHAP
                draw_header("Stage 1 — Visual Explanations")
                y = PH - M - 28
                img3 = to_img_reader(s1.get("crop"))
                img4 = to_img_reader(s1.get("gradcam"))
                img5 = to_img_reader(s1.get("shap"))
                # Crop on left, Grad-CAM on right
                draw_image(img3, M, y - row_h, col_w, row_h)
                draw_image(img4, M + col_w + 12, y - row_h, col_w, row_h)
                c.setFont("Helvetica", 9)
                c.drawString(M, y - row_h - 12, "Cropped ROI")
                c.drawString(M + col_w + 12, y - row_h - 12, "Grad-CAM++")
                # SHAP full width below
                img_w = PW - 2*M
                draw_image(img5, M, M + 30, img_w, row_h)
                c.setFont("Helvetica", 9)
                c.drawString(M, M + 20, "GradientSHAP")
                c.showPage()

            # ----- Stage 2 page -----
            if s2:
                draw_header("Stage 2 — Coimbra (Tabular)")
                y = PH - M - 28
                lines = [
                    f"Probability of malignancy: {_fmt_pct(s2['p_malignant'])}",
                    "Input features (as provided):",
                ]
                y = draw_kv(lines, M, y, leading=14) - 6
                # Print features in 2 columns for better use of space
                items = list(s2["features"].items())
                mid = (len(items)+1)//2
                left, right = items[:mid], items[mid:]
                c.setFont("Helvetica", 9)
                yy = y
                for k, v in left:
                    c.drawString(M, yy, f"- {k}: {v}")
                    yy -= 12
                yy2 = y
                for k, v in right:
                    c.drawString(PW/2, yy2, f"- {k}: {v}")
                    yy2 -= 12
                c.showPage()

            # ----- Stage 3 page -----
            if s3:
                draw_header("Stage 3 — Wisconsin (Tabular)")
                y = PH - M - 28
                lines = [
                    f"Probability of malignancy: {_fmt_pct(s3['p_malignant'])}",
                    "Input features (as provided):",
                ]
                y = draw_kv(lines, M, y, leading=14) - 6
                items = list(s3["features"].items())
                mid = (len(items)+1)//2
                left, right = items[:mid], items[mid:]
                c.setFont("Helvetica", 9)
                yy = y
                for k, v in left:
                    c.drawString(M, yy, f"- {k}: {v}")
                    yy -= 12
                yy2 = y
                for k, v in right:
                    c.drawString(PW/2, yy2, f"- {k}: {v}")
                    yy2 -= 12
                c.showPage()

            # ----- Footer notes -----
            draw_header("Notes")
            y = PH - M - 28
            y = draw_kv([
                "• This report is generated for research/education. It is not a medical device.",
                "• Data are processed in memory and not stored by the application.",
                "• Models are trained on publicly available datasets and require institutional validation.",
            ], M, y, leading=14)
            c.showPage()

            # Finalize and present the PDF
            c.save()
            pdf_bytes.seek(0)

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            st.download_button(
                "Download PDF report",
                data=pdf_bytes.getvalue(),
                file_name=f"overall_assessment_{ts}.pdf",
                mime="application/pdf",
            )

# =============================== Fallback ====================================
else:
    st.error("Unknown selection.")
