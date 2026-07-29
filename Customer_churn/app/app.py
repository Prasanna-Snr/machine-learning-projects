"""
app/app.py
==========
Streamlit app for Customer Churn Prediction.

Run with:
    streamlit run app/app.py

Self-contained - no utils/ dependency.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── paths ────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


# ── load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    try:
        with open(MODELS_DIR / "best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(MODELS_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(MODELS_DIR / "label_encoders.pkl", "rb") as f:
            label_encoders = pickle.load(f)
        with open(MODELS_DIR / "metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        return model, scaler, label_encoders, meta
    except FileNotFoundError as e:
        return None, None, None, str(e)


# ── page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── load model ────────────────────────────────────────────────────────────────
model, scaler, label_encoders, meta = load_artifacts()

if model is None:
    st.error(
        f"Could not load model artifacts from `{MODELS_DIR}`.\n\n"
        f"Run `main.ipynb` first to train and save the models.\n\n{meta}"
    )
    st.stop()

FEATURES = meta["features"]
CAT_COLS = meta["cat_cols"]

# ── sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.markdown(
    "Predict whether a telecom customer is likely to churn "
    "based on their account and service details.\n\n"
    f"**Model:** {meta['best_model']}\n\n"
    f"**Accuracy:** {meta['test_accuracy']*100:.2f}%\n\n"
    f"**ROC AUC:** {meta['test_roc_auc']:.4f}\n\n"
    f"**F1 Score:** {meta['test_f1']:.4f}\n\n"
    "Built by Prasanna Sunuwar"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**All Model Results**")
for name, r in meta["all_results"].items():
    st.sidebar.markdown(
        f"- **{name}**  \n"
        f"  Acc={r['test_acc']*100:.1f}%  ROC={r['test_roc']:.3f}"
    )

# ── main UI ───────────────────────────────────────────────────────────────────
st.title("Customer Churn Prediction")
st.markdown(
    "Fill in the customer details below and click **Predict** "
    "to estimate the probability of churn."
)
st.markdown("---")

# ── input form ─────────────────────────────────────────────────────────────────
st.subheader("Customer Information")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Demographics**")
    gender          = st.selectbox("Gender",          ["Female", "Male"])
    senior_citizen  = st.selectbox("Senior Citizen",  ["No", "Yes"])
    partner         = st.selectbox("Partner",         ["No", "Yes"])
    dependents      = st.selectbox("Dependents",      ["No", "Yes"])

with col2:
    st.markdown("**Account**")
    tenure          = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12, step=1)
    contract        = st.selectbox("Contract",        ["Month-to-month", "One year", "Two year"])
    paperless       = st.selectbox("Paperless Billing", ["No", "Yes"])
    payment_method  = st.selectbox("Payment Method",
                                   ["Electronic check", "Mailed check",
                                    "Bank transfer (automatic)", "Credit card (automatic)"])

with col3:
    st.markdown("**Charges**")
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=65.0, step=0.5)
    total_charges   = st.number_input("Total Charges ($)",   min_value=0.0, max_value=9000.0, value=780.0, step=1.0)

st.markdown("---")
st.subheader("Services")

col4, col5, col6 = st.columns(3)

with col4:
    phone_service   = st.selectbox("Phone Service",    ["No", "Yes"])
    multiple_lines  = st.selectbox("Multiple Lines",   ["No", "No phone service", "Yes"])
    internet_svc    = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

with col5:
    online_security = st.selectbox("Online Security",    ["No", "No internet service", "Yes"])
    online_backup   = st.selectbox("Online Backup",      ["No", "No internet service", "Yes"])
    device_protect  = st.selectbox("Device Protection",  ["No", "No internet service", "Yes"])

with col6:
    tech_support    = st.selectbox("Tech Support",      ["No", "No internet service", "Yes"])
    streaming_tv    = st.selectbox("Streaming TV",      ["No", "No internet service", "Yes"])
    streaming_movies= st.selectbox("Streaming Movies",  ["No", "No internet service", "Yes"])

st.markdown("---")

# ── predict ───────────────────────────────────────────────────────────────────
if st.button("Predict", use_container_width=True, type="primary"):

    # build raw input dict
    raw = {
        "gender":           gender,
        "SeniorCitizen":    1 if senior_citizen == "Yes" else 0,
        "Partner":          partner,
        "Dependents":       dependents,
        "tenure":           tenure,
        "PhoneService":     phone_service,
        "MultipleLines":    multiple_lines,
        "InternetService":  internet_svc,
        "OnlineSecurity":   online_security,
        "OnlineBackup":     online_backup,
        "DeviceProtection": device_protect,
        "TechSupport":      tech_support,
        "StreamingTV":      streaming_tv,
        "StreamingMovies":  streaming_movies,
        "Contract":         contract,
        "PaperlessBilling": paperless,
        "PaymentMethod":    payment_method,
        "MonthlyCharges":   monthly_charges,
        "TotalCharges":     total_charges,
    }

    # encode categoricals
    encoded = raw.copy()
    for col in CAT_COLS:
        le = label_encoders[col]
        encoded[col] = int(le.transform([raw[col]])[0])

    # build feature array in correct order
    sample    = np.array([[encoded[f] for f in FEATURES]])
    sample_sc = scaler.transform(sample)

    pred      = model.predict(sample_sc)[0]
    proba     = model.predict_proba(sample_sc)[0]
    churn_prob = float(proba[1])
    no_churn_prob = float(proba[0])

    # ── result ────────────────────────────────────────────────────────────────
    st.markdown("### Prediction Result")

    if pred == 1:
        color  = "#c62828"
        label  = "Likely to Churn"
    else:
        color  = "#2e7d32"
        label  = "Not Likely to Churn"

    st.markdown(
        f"""
        <div style="
            background: {color};
            color: white;
            padding: 24px;
            border-radius: 12px;
            text-align: center;
            font-size: 28px;
            font-weight: bold;
            margin-bottom: 12px;
        ">
            {label}
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric("Churn Probability",    f"{churn_prob*100:.1f}%")
        st.progress(churn_prob)
    with col_b:
        st.metric("No Churn Probability", f"{no_churn_prob*100:.1f}%")
        st.progress(no_churn_prob)

    # ── input summary ─────────────────────────────────────────────────────────
    with st.expander("Input Summary"):
        summary_df = pd.DataFrame({
            "Feature": list(raw.keys()),
            "Value":   [str(v) for v in raw.values()],
        })
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

else:
    st.info(
        "Fill in the customer details above and click **Predict** to get a churn prediction."
    )
    st.markdown(
        "**Key churn indicators:**\n"
        "- Month-to-month contracts have higher churn\n"
        "- Customers with shorter tenure are more likely to leave\n"
        "- Fiber optic internet users churn more frequently\n"
        "- Customers without online security or tech support churn more"
    )
