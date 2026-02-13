
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("air_quality_model.pkl")
scaler = joblib.load("scaler.pkl")
model_columns = joblib.load("model_columns.pkl")
label_encoder = joblib.load("label_encoder.pkl")

st.title("Air Quality Classification App")

input_data = {}
for col in model_columns:
    input_data[col] = st.number_input(f"Enter {col}", value=0.0)

if st.button("Predict"):
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    result = label_encoder.inverse_transform(prediction)
    st.success(f"Prediction: {result[0]}")
