import os
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model and scaler
@st.cache_resource
def load_resources():
    with open('rf_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('rf_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('label_encoders.pkl', 'rb') as encoders_file:
        encoders = pickle.load(encoders_file)
    return model, scaler, encoders

# Preprocess input features
def preprocess_input(features, scaler):
    try:
        scaled_features = scaler.transform([features])
        return scaled_features
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        return None

# App UI
def main():
    st.title("Plant Disease Prediction")
    st.write("Enter the details to predict the disease.")

    # Load model, scaler, and encoders
    model, scaler, encoders = load_resources()

    # Input fields for plant disease prediction
    plant_name = st.selectbox("Plant Name", encoders['Plant Name'].classes_)
    severity = st.selectbox("Severity", encoders['Severity'].classes_)
    region = st.selectbox("Region", encoders['Region'].classes_)
    days_since_detection = st.number_input("Days Since Detection", min_value=0, step=1)
    treatment_status = st.selectbox("Treatment Status", encoders['Treatment Status'].classes_)

    # Encode categorical inputs
    plant_name_encoded = encoders['Plant Name'].transform([plant_name])[0]
    severity_encoded = encoders['Severity'].transform([severity])[0]
    region_encoded = encoders['Region'].transform([region])[0]
    treatment_status_encoded = encoders['Treatment Status'].transform([treatment_status])[0]

    # Combine inputs into a feature array
    user_input = [plant_name_encoded, severity_encoded, region_encoded, days_since_detection, treatment_status_encoded]

    if st.button("Predict Disease"):
        st.write("### Debug Information")
        st.write("Encoded User Input:", user_input)

        # Preprocess user input
        scaled_input = preprocess_input(user_input, scaler)
        if scaled_input is not None:
            st.write("Scaled Input:", scaled_input)

            # Make prediction
            try:
                prediction = model.predict(scaled_input.reshape(1, -1))[0]
                disease_prediction = encoders['Disease'].inverse_transform([prediction])[0]
                st.success(f"Predicted Disease: {disease_prediction}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
# Reset Button
if st.button("Reset"):
    st.rerun()
