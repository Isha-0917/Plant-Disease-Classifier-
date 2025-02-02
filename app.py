import os
import streamlit as st
import numpy as np
import pickle  # Assuming the model will be saved and loaded using pickle

# Load the pre-trained model and scaler
@st.cache_resource
def load_resources():
    with open('rf_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('rf_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

# Preprocess input features
def preprocess_input(features, scaler):
    try:
        # Apply scaling to the input features
        scaled_features = scaler.transform([features])
        return scaled_features
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        return None

# App UI
def main():
    st.title("Plant Disease Prediction")
    st.write("Enter the details to predict the disease.")

    # Load model and scaler
    model, scaler = load_resources()

    # Input fields for plant disease prediction
    plant_name = st.text_input("Plant Name")
    disease = st.text_input("Disease")
    severity = st.number_input("Severity", min_value=0.0, step=0.1, format="%.2f")
    region = st.text_input("Region")
    days_since_detection = st.number_input("Days Since Detection", min_value=0, step=1)
    treatment_status = st.selectbox("Treatment Status", ["Not Started", "In Progress", "Completed"])

    # Convert categorical inputs into numeric or encoded form if required
    treatment_mapping = {"Not Started": 0, "In Progress": 1, "Completed": 2}
    treatment_status_encoded = treatment_mapping[treatment_status]

    # Combine inputs into a feature array
    user_input = [severity, days_since_detection, treatment_status_encoded]

    if st.button("Predict Disease"):
        st.write("### Debug Information")
        st.write("User Input:", user_input)

        # Preprocess user input
        scaled_input = preprocess_input(user_input, scaler)
        if scaled_input is not None:
            st.write("Scaled Input:", scaled_input)

            # Make prediction
            try:
                prediction = model.predict(scaled_input.reshape(1, -1))[0]
                st.success(f"Predicted Disease: {prediction}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
# Reset Button
if st.button("Reset"):
    st.rerun()
