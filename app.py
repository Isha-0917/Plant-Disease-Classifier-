import os
import streamlit as st
import joblib
import numpy as np

# Load components
try:
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
except Exception as e:
    st.error(f"Error loading essential components: {e}")
    st.stop()

# Try loading the model
try:
    model = joblib.load("rf_classifier.pkl")
    model_loaded = True
except Exception as e:
    st.warning(f"Classifier model could not be loaded: {e}")
    model_loaded = False

# App title
st.title("Plant Disease Classifier")
st.write("Provide the following details to predict the disease.")

# User inputs
plant_name = st.selectbox("Plant Name", label_encoders["Plant Name"].classes_)
disease = st.selectbox("Disease", label_encoders["Disease"].classes_)
severity = st.selectbox("Severity", label_encoders["Severity"].classes_)
region = st.selectbox("Region", label_encoders["Region"].classes_)
treatment_status = st.selectbox(
    "Treatment Status", label_encoders["Treatment Status"].classes_
)
days_since_detection = st.number_input(
    "Days Since Detection", min_value=0, step=1, format="%d"
)

# Process inputs
if st.button("Predict"):
    if not model_loaded:
        st.error("Prediction is unavailable because the classifier model could not be loaded.")
    else:
        try:
            # Encode categorical features
            plant_name_encoded = label_encoders["Plant Name"].transform([plant_name])[0]
            disease_encoded = label_encoders["Disease"].transform([disease])[0]
            severity_encoded = label_encoders["Severity"].transform([severity])[0]
            region_encoded = label_encoders["Region"].transform([region])[0]
            treatment_status_encoded = label_encoders["Treatment Status"].transform(
                [treatment_status]
            )[0]

            # Prepare input array
            features = np.array(
                [
                    plant_name_encoded,
                    disease_encoded,
                    severity_encoded,
                    region_encoded,
                    treatment_status_encoded,
                    days_since_detection,
                ]
            ).reshape(1, -1)

            # Scale features
            features_scaled = scaler.transform(features)

            # Predict
            prediction = model.predict(features_scaled)
            st.success(f"Predicted Disease: {label_encoders['Disease'].inverse_transform(prediction)[0]}")

        except Exception as e:
            st.error(f"Error during prediction: {e}")
# Reset Button
if st.button("Reset"):
    st.rerun()
