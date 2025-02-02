import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model
import os

model= joblib.load('rf_classifier.pkl')
scaler=joblib.load('rf_scaler.pkl')

st.title("Plant🌿 Disease Classifier App")
st.write("Enter the features below to classify the plant disease.")

# Feature Names (Modify as per your dataset)
feature_names = ["Plant Name", "Severity", "Region", "Temperature", "Magnesium"]

# Create input fields dynamically
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, step=0.1, format="%.2f")
    user_input.append(value)

# Predict Button
if st.button("Predict Plant Disease"):
    if None in user_input:
        st.warning(" Please enter all values before predicting.")
    else:
        # Convert input to 2D array and scale it
        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # Display Result
        st.success(f" Predicted Plant Disease: *{prediction[0]}*")
        st.write(f" Prediction Confidence: {max(prediction_proba[0]) * 100:.2f}%")

        # Download Option
        st.download_button("Download Prediction", f"Plant Category: {prediction[0]}", file_name="prediction.txt")

# Reset Button
if st.button("Reset"):
    st.rerun()
