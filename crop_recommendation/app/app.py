"""
app/app.py
==========
Streamlit app for Crop Recommendation.

Run with:
    streamlit run app/app.py
"""

import pickle
from pathlib import Path

import numpy as np
import streamlit as st

# ── paths ───────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


# ── load artifacts ──────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    try:
        with open(MODELS_DIR / "best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(MODELS_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(MODELS_DIR / "label_encoder.pkl", "rb") as f:
            le = pickle.load(f)
        with open(MODELS_DIR / "metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        return model, scaler, le, meta
    except FileNotFoundError as e:
        return None, None, None, str(e)


# ── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Crop Recommendation",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── load model ───────────────────────────────────────────────────────────────
model, scaler, le, meta = load_artifacts()

if model is None:
    st.error(
        f"Could not load model artifacts from `{MODELS_DIR}`.\n\n"
        f"Run `main.ipynb` first to train and save the models.\n\n{meta}"
    )
    st.stop()

FEATURES = meta["features"]
CLASSES  = meta["classes"]

# ── sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.markdown(
    "This app recommends the best crop to grow based on "
    "soil nutrients and environmental conditions.\n\n"
    f"**Model:** {meta['best_model']}\n\n"
    f"**Accuracy:** {meta['test_accuracy']*100:.2f}%\n\n"
    f"**Crops:** {len(CLASSES)}\n\n"
    "Built by Prasanna Sunuwar"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Supported Crops**")
for crop in sorted(CLASSES):
    st.sidebar.markdown(f"- {crop}")

# ── main UI ───────────────────────────────────────────────────────────────────
st.title("Crop Recommendation System")
st.markdown(
    "Enter your **soil** and **climate** values below "
    "to get a personalised crop recommendation."
)
st.markdown("---")

# ── input form ────────────────────────────────────────────────────────────────
st.subheader("Input Parameters")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Soil Nutrients**")
    N           = st.number_input("Nitrogen (N)",     min_value=0.0,   max_value=300.0, value=90.0,  step=0.1,  help="Nitrogen content in soil (kg/ha)")
    P           = st.number_input("Phosphorus (P)",   min_value=0.0,   max_value=300.0, value=42.0,  step=0.1,  help="Phosphorus content in soil (kg/ha)")
    K           = st.number_input("Potassium (K)",    min_value=0.0,   max_value=300.0, value=43.0,  step=0.1,  help="Potassium content in soil (kg/ha)")
    ph          = st.number_input("Soil pH",          min_value=0.0,   max_value=14.0,  value=6.5,   step=0.01, help="pH value of the soil (0-14)")

with col2:
    st.markdown("**Climate Conditions**")
    temperature = st.number_input("Temperature (C)",  min_value=-10.0, max_value=60.0,  value=20.8,  step=0.1,  help="Average temperature in Celsius")
    humidity    = st.number_input("Humidity (%)",     min_value=0.0,   max_value=100.0, value=82.0,  step=0.1,  help="Relative humidity in %")
    rainfall    = st.number_input("Rainfall (mm)",    min_value=0.0,   max_value=800.0, value=202.0, step=0.1,  help="Annual rainfall in mm")

st.markdown("---")

# ── predict ───────────────────────────────────────────────────────────────────
if st.button("Recommend Crop", use_container_width=True, type="primary"):
    sample     = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
    sample_sc  = scaler.transform(sample)

    pred_idx   = model.predict(sample_sc)[0]
    proba      = model.predict_proba(sample_sc)[0]
    crop       = le.inverse_transform([pred_idx])[0]
    confidence = float(proba[pred_idx])

    # ── result card ────────────────────────────────────────────────────────
    st.markdown("### Recommendation")
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #2e7d32, #66bb6a);
            color: white;
            padding: 28px;
            border-radius: 16px;
            text-align: center;
            font-size: 36px;
            font-weight: bold;
            margin-bottom: 12px;
        ">
            {crop}
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(f"**Confidence:** `{confidence:.1%}`")
    st.progress(float(confidence))

    # ── top-5 probabilities ────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Top 5 Crop Probabilities")

    top5_idx   = np.argsort(proba)[::-1][:5]
    top5_crops = le.inverse_transform(top5_idx)
    top5_probs = proba[top5_idx]

    for c, p in zip(top5_crops, top5_probs):
        st.markdown(f"**{c}**")
        st.progress(float(p))
        st.caption(f"{p:.1%}")

    # ── input summary ──────────────────────────────────────────────────────
    with st.expander("Input Summary"):
        import pandas as pd
        summary = pd.DataFrame({
            "Feature": FEATURES,
            "Value":   [N, P, K, temperature, humidity, ph, rainfall],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

else:
    st.info(
        "Fill in the soil and climate values above, "
        "then click **Recommend Crop** to get a prediction."
    )
    st.markdown(
        "**Feature guide:**\n"
        "- **N / P / K** — soil macronutrients (kg/ha)\n"
        "- **pH** — soil acidity (6-7 is neutral)\n"
        "- **Temperature** — average Celsius for the growing season\n"
        "- **Humidity** — relative humidity (%)\n"
        "- **Rainfall** — annual rainfall (mm)"
    )
