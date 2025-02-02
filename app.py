import streamlit as st
import pickle
import numpy as np

# Load the pre-trained model, scaler, and label encoders
@st.cache_resource
def load_resources():
    with open('rf_classifier.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('rf_scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('label_encoders.pkl', 'rb') as encoder_file:
        label_encoders = pickle.load(encoder_file)
    return model, scaler, label_encoders

# Preprocess input features
def preprocess_input(features, scaler):
    scaled_features = scaler.transform([features])
    return scaled_features

# App UI
def main():
    st.title("Plant Disease Classifier")
    st.write("Enter the features of the plant to predict the disease category.")

    # Example feature input fields (replace with actual feature names)
    feature_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]

    user_input = []
    for feature in feature_names:
        value = st.number_input(f"{feature}", min_value=0.0, step=0.1, format="%.2f")
        user_input.append(value)

    if st.button("Predict Disease"):
        model, scaler, label_encoders = load_resources()

        # Preprocess user input
        scaled_input = preprocess_input(user_input, scaler)

        # Make prediction
        prediction = model.predict(scaled_input)[0]
        prediction_label = label_encoders.inverse_transform([prediction])[0]

        # Display result
        st.success(f"Predicted Disease: {prediction_label}")

if __name__ == "__main__":
    main()
