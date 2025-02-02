import os
import streamlit as st
import joblib
import numpy as np

# Load the required components
try:
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    model = joblib.load("rf_classifier.pkl")
except Exception as e:
    st.error(f"Error loading required files: {e}")
    st.stop()

# App title
st.title("Plant Disease Classifier")
st.write("Enter the plant details to predict the disease.")

# Input features
plant_name = st.selectbox("Plant Name", label_encoders["Plant Name"].classes_)
severity = st.selectbox("Severity", label_encoders["Severity"].classes_)
region = st.selectbox("Region", label_encoders["Region"].classes_)
treatment_status = st.selectbox(
    "Treatment Status", label_encoders["Treatment Status"].classes_
)
days_since_detection = st.number_input(
    "Days Since Detection", min_value=0, step=1, format="%d"
)

# Predict button
if st.button("Predict"):
    try:
        # Encode categorical features
        plant_name_encoded = label_encoders["Plant Name"].transform([plant_name])[0]
        severity_encoded = label_encoders["Severity"].transform([severity])[0]
        region_encoded = label_encoders["Region"].transform([region])[0]
        treatment_status_encoded = label_encoders["Treatment Status"].transform(
            [treatment_status]
        )[0]

        # Prepare feature array
        features = np.array(
            [
                plant_name_encoded,
                severity_encoded,
                region_encoded,
                treatment_status_encoded,
                days_since_detection,
            ]
        ).reshape(1, -1)

        # Scale features
        features_scaled = scaler.transform(features)

        # Predict disease
        prediction = model.predict(features_scaled)
        predicted_disease = label_encoders["Disease"].inverse_transform(prediction)[0]

        # Display the prediction
        st.success(f"Predicted Disease: {predicted_disease}")
    except Exception as e:
        st.error("An error occurred during prediction. Please try again.")
# Reset Button
if st.button("Reset"):
    st.rerun()
