"""
app/app.py
==========
Streamlit app for Walmart Sales Forecasting.

Run with:
    streamlit run app/app.py

Self-contained - no src/ dependency.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── paths ─────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"


# ── load artifacts ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model...")
def load_artifacts():
    try:
        with open(MODELS_DIR / "best_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open(MODELS_DIR / "scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open(MODELS_DIR / "metadata.pkl", "rb") as f:
            meta = pickle.load(f)
        return model, scaler, meta
    except FileNotFoundError as e:
        return None, None, str(e)


# ── page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ── load ───────────────────────────────────────────────────────────────────────
model, scaler, meta = load_artifacts()

if model is None:
    st.error(
        f"Could not load model artifacts from `{MODELS_DIR}`.\n\n"
        f"Run `main.ipynb` first to train and save the models.\n\n{meta}"
    )
    st.stop()

FEATURES = meta["features"]
STORES   = meta["stores"]

# ── sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.title("About")
st.sidebar.markdown(
    "Predict weekly sales for a Walmart store based on "
    "operational and economic conditions.\n\n"
    f"**Model:** {meta['best_model']}\n\n"
    f"**R2 Score:** {meta['r2']:.4f}\n\n"
    f"**MAE:** ${meta['mae']:,.0f}\n\n"
    f"**RMSE:** ${meta['rmse']:,.0f}\n\n"
    "Built by Prasanna Sunuwar"
)
st.sidebar.markdown("---")
st.sidebar.markdown("**All Model Results**")
for name, r in meta["all_results"].items():
    st.sidebar.markdown(
        f"- **{name}**  \n"
        f"  R2={r['r2']:.3f}  MAE=${r['mae']:,.0f}"
    )

# ── main UI ────────────────────────────────────────────────────────────────────
st.title("Walmart Weekly Sales Forecast")
st.markdown(
    "Enter the store details and economic indicators below, "
    "then click **Predict** to get a weekly sales forecast."
)
st.markdown("---")

# ── input form ─────────────────────────────────────────────────────────────────
st.subheader("Input Features")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Store and Date**")
    store        = st.selectbox("Store Number", options=STORES, index=0)
    date_input   = st.date_input("Forecast Date", value=pd.Timestamp("2012-11-02"))
    holiday_flag = st.selectbox("Holiday Week", options=["No", "Yes"])

    st.markdown("**Economic Indicators**")
    temperature  = st.number_input("Temperature (F)", min_value=-10.0, max_value=120.0, value=60.0, step=0.5)
    fuel_price   = st.number_input("Fuel Price ($/gallon)", min_value=1.0, max_value=6.0, value=3.0, step=0.01)
    cpi          = st.number_input("CPI", min_value=100.0, max_value=280.0, value=211.0, step=0.1)
    unemployment = st.number_input("Unemployment Rate (%)", min_value=3.0, max_value=15.0, value=8.0, step=0.1)

with col2:
    st.markdown("**Historical Sales (for lag features)**")
    lag_1 = st.number_input("Last Week Sales - Lag 1 ($)",    min_value=0.0, value=1_500_000.0, step=1000.0)
    lag_2 = st.number_input("2 Weeks Ago Sales - Lag 2 ($)",  min_value=0.0, value=1_480_000.0, step=1000.0)
    lag_4 = st.number_input("4 Weeks Ago Sales - Lag 4 ($)",  min_value=0.0, value=1_460_000.0, step=1000.0)
    lag_12= st.number_input("12 Weeks Ago Sales - Lag 12 ($)",min_value=0.0, value=1_400_000.0, step=1000.0)

st.markdown("---")

# ── predict ────────────────────────────────────────────────────────────────────
if st.button("Predict Weekly Sales", use_container_width=True, type="primary"):

    # derive date features
    date        = pd.Timestamp(date_input)
    year        = date.year
    month       = date.month
    quarter     = date.quarter
    week        = int(date.isocalendar().week)
    day         = date.day
    day_of_week = date.dayofweek
    is_weekend  = int(day_of_week >= 5)
    hf          = 1 if holiday_flag == "Yes" else 0

    # approximate rolling features from lag inputs
    recent = [lag_1, lag_2, lag_4]
    rolling_mean_4  = np.mean(recent)
    rolling_std_4   = float(np.std(recent)) if len(recent) > 1 else lag_1 * 0.05
    rolling_max_4   = float(np.max(recent))
    rolling_min_4   = float(np.min(recent))

    all_lags = [lag_1, lag_2, lag_4, lag_12]
    rolling_mean_12 = np.mean(all_lags)
    rolling_std_12  = float(np.std(all_lags)) if len(all_lags) > 1 else lag_1 * 0.07
    rolling_max_12  = float(np.max(all_lags))
    rolling_min_12  = float(np.min(all_lags))
    expanding_mean  = np.mean(all_lags)

    # build feature vector in training order
    feature_map = {
        "Store":            store,
        "Holiday_Flag":     hf,
        "Temperature":      temperature,
        "Fuel_Price":       fuel_price,
        "CPI":              cpi,
        "Unemployment":     unemployment,
        "Year":             year,
        "Month":            month,
        "Quarter":          quarter,
        "Week":             week,
        "Day":              day,
        "DayOfWeek":        day_of_week,
        "IsWeekend":        is_weekend,
        "Lag_1":            lag_1,
        "Lag_2":            lag_2,
        "Lag_4":            lag_4,
        "Lag_12":           lag_12,
        "Rolling_Mean_4":   rolling_mean_4,
        "Rolling_Std_4":    rolling_std_4,
        "Rolling_Max_4":    rolling_max_4,
        "Rolling_Min_4":    rolling_min_4,
        "Rolling_Mean_12":  rolling_mean_12,
        "Rolling_Std_12":   rolling_std_12,
        "Rolling_Max_12":   rolling_max_12,
        "Rolling_Min_12":   rolling_min_12,
        "Expanding_Mean":   expanding_mean,
    }

    sample    = np.array([[feature_map[f] for f in FEATURES]])
    sample_sc = scaler.transform(sample)
    pred      = float(model.predict(sample_sc)[0])
    pred      = max(0.0, pred)

    # result
    st.markdown("### Prediction Result")
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #1565c0, #42a5f5);
            color: white;
            padding: 24px;
            border-radius: 12px;
            text-align: center;
            font-size: 32px;
            font-weight: bold;
            margin-bottom: 12px;
        ">
            ${pred:,.0f}
        </div>
        """,
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Store",        f"Store {store}")
    c2.metric("Holiday Week", holiday_flag)
    c3.metric("Week",         f"Week {week}, {year}")

    # input summary
    with st.expander("Input Summary"):
        summary = pd.DataFrame({
            "Feature": ["Store", "Date", "Holiday", "Temperature (F)",
                        "Fuel Price ($/gal)", "CPI", "Unemployment (%)",
                        "Lag 1 ($)", "Lag 2 ($)", "Lag 4 ($)", "Lag 12 ($)"],
            "Value": [store, str(date_input), holiday_flag,
                      temperature, fuel_price, cpi, unemployment,
                      f"${lag_1:,.0f}", f"${lag_2:,.0f}",
                      f"${lag_4:,.0f}", f"${lag_12:,.0f}"],
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

else:
    st.info(
        "Fill in the store details and economic indicators above, "
        "then click **Predict Weekly Sales**."
    )
    st.markdown(
        "**Feature guide:**\n"
        "- **Store** — Walmart store number (1-45)\n"
        "- **Holiday Week** — whether the week contains a major holiday\n"
        "- **Temperature** — average temperature in Fahrenheit\n"
        "- **Fuel Price** — regional fuel price in $/gallon\n"
        "- **CPI** — Consumer Price Index\n"
        "- **Unemployment** — regional unemployment rate\n"
        "- **Lag features** — historical weekly sales for that store"
    )
