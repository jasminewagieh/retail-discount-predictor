import streamlit as st
import joblib
import pandas as pd

# Load model
model = joblib.load("discount_model.joblib")

st.title("Retail Discount Predictor")

# User input
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
# ... add all required features

if st.button("Predict"):
    input_df = pd.DataFrame([[feature1, feature2]], columns=["Feature1", "Feature2"])
    prediction = model.predict(input_df)
    st.write("Prediction:", prediction[0])
