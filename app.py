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
        # If scaler is a NumPy array, assume it contains scaling factors (standardization)
        if isinstance(scaler, np.ndarray):
            scaled_features = (features - scaler.mean(axis=0)) / scaler.std(axis=0)
        else:
            scaled_features = scaler.transform([features])
        return scaled_features
    except Exception as e:
        st.error(f"Error during scaling: {e}")
        return None

# App UI
def main():
    st.title("Random Forest Classifier")
    st.write("Enter the features to predict the class.")

    # Load model and scaler
    model, scaler = load_resources()

    # Determine number of features dynamically
    if isinstance(scaler, np.ndarray):
        num_features = scaler.shape[0]
    else:
        num_features = scaler.mean_.shape[0]

    # Generate placeholder feature names
    feature_names = [f"Feature {i+1}" for i in range(num_features)]

    # Collect user input
    user_input = []
    for feature in feature_names:
        value = st.number_input(f"{feature}", min_value=0.0, step=0.1, format="%.2f")
        user_input.append(value)

    # Predict button
    if st.button("Predict"):
        st.write("### Debug Information")
        st.write("User Input:", user_input)

        # Preprocess user input
        scaled_input = preprocess_input(np.array(user_input), scaler)
        if scaled_input is not None:
            st.write("Scaled Input:", scaled_input)

            # Make prediction
            try:
                prediction = model.predict(scaled_input.reshape(1, -1))[0]
                st.success(f"Predicted Class: {prediction}")
            except Exception as e:
                st.error(f"Error during prediction: {e}")

if __name__ == "__main__":
    main()
# Reset Button
if st.button("Reset"):
    st.rerun()
