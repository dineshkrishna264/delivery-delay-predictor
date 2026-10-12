import os
import joblib
import pandas as pd
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Olist Delay Risk Predictor",
    page_icon="📦",
    layout="wide",
)

st.title("📦 E-Commerce Delivery Delay Risk Evaluator")
st.markdown(
    "Interactively adjust order parameters below for instant real-time shipping delay evaluation."
)

st.divider()


# Cache model loading into memory so it only loads once across reruns
@st.cache_resource
def load_model_artifact():
    model_path = os.path.join("models", "delay_model.joblib")
    if not os.path.exists(model_path):
        st.error("❌ Model artifact not found at `models/delay_model.joblib`.")
        st.stop()
    return joblib.load(model_path)


artifact = load_model_artifact()

# Interactive Input Layout (2 Columns)
col1, col2 = st.columns(2)

with col1:
    st.subheader("📍 Order & Spatial Specs")
    distance_km = st.slider("Shipping Distance (km)", 10.0, 3000.0, 450.0)
    is_same_state = st.radio(
        "Shipping Route",
        options=[1, 0],
        format_func=lambda x: (
            "Local (Same State)" if x == 1 else "Interstate (Cross-Border)"
        ),
    )
    estimated_days = st.slider("Promised Delivery Window (Days)", 1, 45, 14)

    st.subheader("🕒 Operational Timing")
    purchase_dow = st.selectbox(
        "Day of Week Purchased",
        options=[0, 1, 2, 3, 4, 5, 6],
        format_func=lambda x: [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ][x],
    )
    purchase_hour = st.slider("Hour of Purchase (0-23)", 0, 23, 14)

with col2:
    st.subheader("💰 Financial & Physical Cargo Traits")
    total_freight_value = st.number_input(
        "Total Freight Charge (BRL)", 0.0, 500.0, 18.50
    )
    total_item_price = st.number_input(
        "Total Cart Item Price (BRL)", 0.0, 5000.0, 120.00
    )
    num_items = st.number_input("Number of Items", 1, 20, 1)
    product_weight_g = st.number_input(
        "Package Weight (Grams)", 50.0, 30000.0, 800.0
    )
    product_volume_cm3 = st.number_input(
        "Package Volume (cm³)", 100.0, 100000.0, 3000.0
    )

st.divider()

# Construct Input DataFrame matching exact model feature names
payload = {
    "total_freight_value": total_freight_value,
    "total_item_price": total_item_price,
    "num_items": int(num_items),
    "product_weight_g": product_weight_g,
    "product_volume_cm3": product_volume_cm3,
    "distance_km": distance_km,
    "is_same_state": int(is_same_state),
    "estimated_days": estimated_days,
    "purchase_dow": purchase_dow,
    "purchase_hour": purchase_hour,
}

df_input = pd.DataFrame([payload])[artifact["feature_names"]]

# Instant Live Inference
prob = float(artifact["model"].predict_proba(df_input)[0, 1])

if prob < 0.25:
    risk_level = "LOW"
elif prob < 0.45:
    risk_level = "MEDIUM"
else:
    risk_level = "HIGH"

# Real-Time Visual Display
st.subheader("⚡ Live Risk Evaluation Result")
res_col1, res_col2 = st.columns([1, 2])

with res_col1:
    st.metric(label="Calculated Delay Risk", value=f"{prob * 100:.1f}%")

with res_col2:
    if risk_level == "HIGH":
        st.error(
            "⚠️ **HIGH DELAY RISK DETECTED**\n\nThis package requires priority carrier assignment or proactive customer delivery notification."
        )
    elif risk_level == "MEDIUM":
        st.warning(
            "⚡ **MODERATE DELAY RISK**\n\nMonitor fulfillment SLA times closely."
        )
    else:
        st.success(
            "✅ **LOW DELAY RISK**\n\nOrder parameters align with standard on-time delivery schedules."
        )