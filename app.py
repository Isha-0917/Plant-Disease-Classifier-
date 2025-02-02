import os
import streamlit as st
import joblib
import numpy as np

# Load components
try:
    scaler = joblib.load("scaler.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
    model = joblib.load("rf_classifier.pkl")
except Exception as e:
    st.error(f"Error loading essential components: {e}")
    st.stop()

# App title
st.title("Plant Disease Classifier")
st.write(
    "Provide the following details to validate or predict the disease affecting the plant."
)

# User inputs
plant_name = st.selectbox("Plant Name", label_encoders["Plant Name"].classes_)
disease_input = st.selectbox("Suspected Disease", label_encoders["Disease"].classes_)
severity = st.selectbox("Severity", label_encoders["Severity"].classes_)
region = st.selectbox("Region", label_encoders["Region"].classes_)
treatment_status = st.selectbox(
    "Treatment Status", label_encoders["Treatment Status"].classes_
)
days_since_detection = st.number_input(
    "Days Since Detection", min_value=0, step=1, format="%d"
)

# Process inputs
if st.button("Predict and Validate"):
    try:
        # Encode categorical features
        plant_name_encoded = label_encoders["Plant Name"].transform([plant_name])[0]
        disease_encoded = label_encoders["Disease"].transform([disease_input])[0]
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

        # Decode predicted disease
        predicted_disease = label_encoders["Disease"].inverse_transform(prediction)[0]

        # Compare user input with model prediction
        if disease_input == predicted_disease:
            st.success(f"The model agrees with your input. Predicted Disease: {predicted_disease}")
        else:
            st.warning(
                f"The model predicts a different disease. "
                f"Input Disease: {disease_input}, Predicted Disease: {predicted_disease}"
            )

    except Exception as e:
        st.error(f"Unexpected error during prediction: {e}")
        st.write("Debugging Information:")
        st.write(f"Features: {features}")
        st.write(f"Scaled Features: {features_scaled}")
# Reset Button
if st.button("Reset"):
    st.rerun()
