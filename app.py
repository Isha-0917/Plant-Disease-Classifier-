import streamlit as st
import pickle
import numpy as np

# Load the scaler and classifier
scaler_path = "rf_scaler.pkl"
classifier_path = "rf_classifier.pkl"

with open(scaler_path, 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

with open(classifier_path, 'rb') as classifier_file:
    classifier = pickle.load(classifier_file)

# Title and description
st.title("Random Forest Classifier")
st.write("Upload input features to make predictions using the trained Random Forest Classifier.")

# Input section
def user_input_features():
    st.sidebar.header("Input Features")
    
    # Replace these placeholders with actual feature names
    feature1 = st.sidebar.number_input("Feature 1", min_value=0.0, max_value=100.0, value=50.0)
    feature2 = st.sidebar.number_input("Feature 2", min_value=0.0, max_value=100.0, value=50.0)
    feature3 = st.sidebar.number_input("Feature 3", min_value=0.0, max_value=100.0, value=50.0)
    feature4 = st.sidebar.number_input("Feature 4", min_value=0.0, max_value=100.0, value=50.0)

    # Continue adding input fields for all required features
    data = {
        "Feature 1": feature1,
        "Feature 2": feature2,
        "Feature 3": feature3,
        "Feature 4": feature4,
    }
    
    return np.array(list(data.values())).reshape(1, -1)

# Get input from the user
input_features = user_input_features()

# Scale the input and make predictions
scaled_features = scaler.transform(input_features)
prediction = classifier.predict(scaled_features)

# Display the prediction
st.subheader("Prediction")
st.write(f"The model predicts: {prediction[0]}")
