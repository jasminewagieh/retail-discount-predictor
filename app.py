import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Retail Discount Predictor", layout="centered")
st.title("Retail II — Discount Prediction")

# load artifacts
model = joblib.load("discount_model.joblib")
cols  = joblib.load("model_columns.joblib")   # list of training column names

# --- inputs ---
qty   = st.number_input("Quantity", min_value=1, value=3)
price = st.number_input("Price (£/unit)", min_value=0.01, value=5.00, step=0.01)
hour  = st.slider("Hour of Day", 0, 23, 12)
month = st.slider("Month", 1, 12, 6)
dow   = st.selectbox("Day of Week", ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"])

# build row exactly like training
row = pd.DataFrame([{"Quantity": qty, "Price": price, "Hour": hour, "Month": month, "DayOfWeek": dow}])
row = pd.get_dummies(row, columns=["DayOfWeek"], drop_first=True)

# align to training columns (creates any missing one-hot columns and orders them)
row = row.reindex(columns=cols, fill_value=0)

# --- DEBUG (optional): show columns app vs expected ---
with st.expander("Debug: columns"):
    st.write("Incoming columns:", list(row.columns))
    st.write("Expected columns:", cols)

if st.button("Predict"):
    proba = model.predict_proba(row)[0, 1]
    pred  = "Discount" if proba >= 0.5 else "No Discount"
    st.metric("Prediction", pred)
    st.write(f"Probability of Discount: **{proba:.2%}**")
