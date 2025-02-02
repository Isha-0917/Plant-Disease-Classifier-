import streamlit as st
import joblib
import numpy as np

# Load the trained Random Forest model
model = joblib.load('rf_classifier.pkl')
scaler = joblib.load('rf_scaler.pkl')

st.title("Wine Classifier App")
st.write("Enter the wine features below to classify the wine category.")

# Feature Names (Modify as per your dataset)
feature_names = ["Feature 1", "Feature 2", "Feature 3", "Feature 4"]

# Create input fields dynamically
user_input = []
for feature in feature_names:
    value = st.number_input(f"{feature}", min_value=0.0, step=0.1, format="%.2f")
    user_input.append(value)

# Predict Button
if st.button("Predict Category"):
    if None in user_input:
        st.warning("Please enter all values before predicting.")
    else:
        # Convert input to 2D array and scale it
        input_array = np.array(user_input).reshape(1, -1)
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)
        prediction_proba = model.predict_proba(scaled_input)

        # Display Result
        st.success(f"Predicted Category: *{prediction[0]}*")
        st.write(f"Prediction Confidence: {max(prediction_proba[0]) * 100:.2f}%")

        # Download Option
        st.download_button("Download Prediction", f"Category: {prediction[0]}", file_name="prediction.txt")

# Reset Button
if st.button("Reset"):
    st.experimental_rerun()
