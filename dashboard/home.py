# Home.py
import streamlit as st
import pandas as pd

# Configure the Streamlit page (wide layout helps with tables and metrics)
st.set_page_config(page_title="Breast Cancer AI Suite", layout="wide")

# """
# Breast Cancer AI Suite — Home Page

# Purpose:
#     This Streamlit page is the landing/home screen for my multi-stage
#     breast cancer AI demo. It:
#       - Shows an ethics/usage notice (research/education only).
#       - Lets users acknowledge the policy before proceeding.
#       - Provides a collapsible "Model Performance" panel with internal metrics.
#       - Routes to two workflows: Early Detection and Full Diagnosis.

# Notes:
#     * This app is not a medical device and must not be used for clinical decisions.
#     * I keep state with st.session_state to preserve toggles and navigation intents.
#     * Navigation uses st.switch_page (with a fallback hint when not available).
# """



# -------------------- session state --------------------
# I use session_state to store UI toggles so they persist across reruns.
if "policy_ack" not in st.session_state:
    st.session_state["policy_ack"] = False  # user must acknowledge policy before proceeding
if "show_perf" not in st.session_state:
    st.session_state["show_perf"] = False   # toggle to reveal/hide the performance panel

# -------------------- title & notice --------------------
st.title("Breast Cancer AI Suite")
st.caption("For research and education only. Not a medical device.")

# I wrap the policy/notice in a bordered container to visually separate it.
with st.container(border=True):
    st.subheader("Important Notice — Research / Educational Use")

    # Clear, explicit policy text. I keep this as Markdown so I can emphasize key points.
    st.markdown(
        """
- This application is intended **solely for research and educational purposes**.  
- It **does not** provide medical advice and **must not** be used as the sole basis for diagnosis, treatment, or patient management.  
- Outputs are **probabilistic** and require **clinical interpretation** by qualified healthcare professionals.  
- The models **must be validated and approved** by the appropriate healthcare organization/regulatory bodies before any clinical deployment.

**Data handling & AI policy (healthcare context)**
- Images and tabular data are **processed in memory** and are **not stored** by the app.  
- Do **not** upload patient identifiers; ensure all data are **de-identified**.  
- Models here were **trained on publicly available datasets** (e.g., CBIS-DDSM, Coimbra, Wisconsin) to demonstrate methodology.  
- Use implies adherence to local **governance, ethics, and data-protection policies** (e.g., IRB, HIPAA/GDPR as applicable).
        """
    )

    # Two side-by-side buttons:
    #   - Acknowledge policy
    #   - Toggle performance view
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button(
            "Acknowledge & Continue",
            type="primary",
            use_container_width=True,
            disabled=st.session_state["policy_ack"],
        ):
            # Once the user acknowledges, I persist it and rerun to update UI state.
            st.session_state["policy_ack"] = True
            st.rerun()
    with c2:
        # Performance panel toggle is available anytime.
        # I keep it separate from policy so it can be inspected before proceeding.
        if st.button(
            "📈 View Model Performance",
            use_container_width=True,
        ):
            st.session_state["show_perf"] = not st.session_state["show_perf"]
            st.rerun()

# -------------------- performance panel --------------------
# When enabled, I present concise, structured metrics for each stage/model.
if st.session_state["show_perf"]:
    st.markdown("### Model Performance (Internal Evaluation)")
    st.caption("Reported metrics are retrospective evaluations and require external validation before any clinical use.")

    # ===== Data I provide (structured for display) =====
    # CRC Lifestyle/Diet (RandomForest) — illustrative configuration and summary
    crc_best = {'max_depth': None, 'max_features': 'sqrt', 'min_samples_leaf': 5, 'n_estimators': 300}
    crc_summary = dict(
        Accuracy=0.88, Precision=0.5714, Sensitivity=0.9032, Specificity=0.8757,
        F1=0.7000, ROC_AUC=0.9505
    )
    crc_report = """\
Benign vs Malignant
precision  recall  f1-score  support
Benign       0.98    0.88     0.93     169
Malignant    0.57    0.90     0.70      31
accuracy                         0.88     200
"""

    # Wisconsin (XGBoost) — headline metrics and a compact text report
    wisc_summary = dict(
        Accuracy=0.9737, ROC_AUC=0.9940
    )
    wisc_report = """\
Benign vs Malignant
precision  recall  f1-score  support
Benign       0.96    1.00     0.98      72
Malignant    1.00    0.93     0.96      42
accuracy                         0.97     114
"""

    # Coimbra (SVM) — I show both train CM and a small validation report for transparency
    coi_train_cm = pd.DataFrame([[45, 2],[2, 55]], index=["True No Cancer","True Cancer"], columns=["Pred No Cancer","Pred Cancer"])
    coi_val_cm   = pd.DataFrame([[4, 1],[3, 4]],   index=["True No Cancer","True Cancer"], columns=["Pred No Cancer","Pred Cancer"])
    coi_train_summary = dict(
        Accuracy=0.9615,  # as reported (training accuracy 96.15%)
        Precision=0.8000, Sensitivity=0.5710, Specificity=0.8000, F1=0.6670, ROC_AUC=0.7710
    )
    coi_val_text = """\
Validation classification report
No Cancer: precision 0.57, recall 0.80, f1 0.67, support 5
Cancer   : precision 0.80, recall 0.57, f1 0.67, support 7
Accuracy 0.67 (8/12)
"""

    # YOLO (Detection) — core detection metrics and an approximate latency breakdown
    yolo_summary = {
        "Precision (B)": 0.8305, "Recall (B)": 0.7050,
        "mAP@50 (B)": 0.8043, "mAP@50-95 (B)": 0.5221, "Fitness": 0.5503
    }
    yolo_speed = {"preprocess (ms/img)": 0.347, "inference (ms/img)": 3.175, "postprocess (ms/img)": 2.069}

    # MaxViT Tiny (Classifier) — operating-point metrics + confusion matrix
    maxvit_summary = dict(
        Threshold=0.753, Accuracy=0.8851, Balanced_Acc=0.8429, Sensitivity=0.7500,
        Specificity=0.9359, Precision=0.8148, NPV=0.9087, F1=0.7811, AUC=0.8957, MCC=0.7045
    )
    maxvit_cm = pd.DataFrame([[219,15],[22,66]], index=["True benign","True malignant"], columns=["Pred benign","Pred malignant"])
    maxvit_report = """\
benign vs malignant (support: 234 / 88)
precision  recall  f1-score
benign      0.91    0.94     0.92
malignant   0.81    0.75     0.78
overall accuracy 0.89 (322 cases)
"""

    # ===== Tabbed view to keep content tidy and scannable =====
    tabs = st.tabs([
        "Stage 1 — Mammogram: YOLO (Detection)",
        "Stage 1 — Mammogram: MaxViT (Classification)",
        "Stage 2 — Coimbra (SVM)",
        "Stage 3 — Wisconsin (XGBoost)",
        "CRC Lifestyle/Diet (RandomForest)"
    ])

    # ---- YOLO tab ----
    with tabs[0]:
        st.subheader("Stage 1 — Mammogram Scanning: YOLO Detector")
        c1, c2 = st.columns(2)
        with c1:
            st.write("**Summary metrics**")
            st.table(pd.DataFrame(yolo_summary, index=["value"]).T)
        with c2:
            st.write("**Throughput (approx.)**")
            st.table(pd.DataFrame(yolo_speed, index=["value"]).T)
        st.caption("Metrics reported on the internal evaluation set; thresholds as per training configuration.")

    # ---- MaxViT tab ----
    with tabs[1]:
        st.subheader("Stage 1 — Mammogram Scanning: MaxViT Classifier")
        st.write("**Metrics at the operating threshold**")
        st.table(pd.DataFrame(maxvit_summary, index=["value"]).T)
        st.write("**Confusion matrix**")
        st.table(maxvit_cm)
        st.text(maxvit_report)

    # ---- Coimbra tab ----
    with tabs[2]:
        st.subheader("Stage 2 — Coimbra (Support Vector Machine)")
        st.write("**Training confusion matrix**")
        st.table(coi_train_cm)
        st.write("**Training summary (medical metrics)**")
        st.table(pd.DataFrame(coi_train_summary, index=["value"]).T)
        st.text(coi_val_text)

    # ---- Wisconsin tab ----
    with tabs[3]:
        st.subheader("Stage 3 — Wisconsin (XGBoost)")
        st.table(pd.DataFrame(wisc_summary, index=["value"]).T)
        st.text(wisc_report)

    # ---- CRC tab ----
    with tabs[4]:
        st.subheader("Colorectal — Dietary & Lifestyle (Random Forest)")
        st.write("**Hyperparameter selection (GridSearchCV)**")
        st.json(crc_best)
        st.write("**Operating threshold**: mode=`target_recall`, selected threshold=0.280")
        st.table(pd.DataFrame(crc_summary, index=["value"]).T)
        st.text(crc_report)

# -------------------- main navigation --------------------
# A divider to separate the performance block from navigation.
st.markdown("---")
st.markdown(
    """
### Select a workflow
- **Early Detection** — tabular risk assessment (single case or CSV).  
- **Full Diagnosis** — Stage 1 (mammography), Stage 2 (Coimbra), Stage 3 (Wisconsin).
"""
)

# I disable navigation until the user acknowledges the policy.
disabled = not st.session_state["policy_ack"]
col1, col2 = st.columns(2)

with col1:
    if st.button("🔬 Early Detection", use_container_width=True, type="primary", disabled=disabled):
        try:
            # Preferred navigation (Streamlit multipage apps)
            st.switch_page("pages/1_Early_Detection.py")
        except Exception:
            # Fallback hint mechanism if switch_page is unavailable (e.g., in some runtimes)
            st.session_state["_goto_page"] = "early"

with col2:
    if st.button("🧠 Full Diagnosis", use_container_width=True, disabled=disabled):
        try:
            st.switch_page("pages/2_Full_Diagnosis.py")
        except Exception:
            st.session_state["_goto_page"] = "full"

# Fallback UX hints when switch_page isn't supported by the environment.
if st.session_state.get("_goto_page") == "early":
    st.info("If the button didn’t navigate, open **Early Detection** from the left sidebar.")
elif st.session_state.get("_goto_page") == "full":
    st.info("If the button didn’t navigate, open **Full Diagnosis** from the left sidebar.")

# Gentle reminder to acknowledge the policy before proceeding.
if not st.session_state["policy_ack"]:
    st.warning("Please review the notice above and click **Acknowledge & Continue** to proceed.")
