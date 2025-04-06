import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# App configuration
st.set_page_config(page_title="Plant Disease Yield Impact Predictor", layout="centered")

# Title
st.title("🌾 Plant Disease Yield Impact Predictor")
st.markdown("Predict how plant diseases affect crop yield using machine learning.")

# Load dataset
@st.cache_data
def load_data():
    data = {
        "Crop Type": ["Wheat", "Wheat", "Rice", "Rice", "Corn", "Corn"],
        "Disease Type": ["Rust", "Blight", "Blight", "Rust", "Wilt", "Blight"],
        "Severity": ["High", "Medium", "Low", "High", "Medium", "Low"],
        "Temperature": [30, 28, 26, 32, 29, 27],
        "Humidity": [80, 75, 70, 85, 78, 72],
        "Yield Impact (%)": [40, 30, 10, 45, 25, 15]
    }
    return pd.DataFrame(data)

df = load_data()

# Show dataset
with st.expander("📊 View Sample Dataset"):
    st.dataframe(df)

# Feature engineering
X = df.drop("Yield Impact (%)", axis=1)
y = df["Yield Impact (%)"]

# Encode categorical variables
encoders = {}
for col in X.select_dtypes(include="object"):
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

# Scale and split
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.subheader("📈 Model Performance")
col1, col2 = st.columns(2)
col1.metric("Mean Absolute Error", f"{mae:.2f}")
col2.metric("R² Score (Accuracy)", f"{r2:.2%}")

# Input section
st.subheader("🧪 Try a Prediction")

crop = st.selectbox("Select Crop Type", df["Crop Type"].unique())
disease = st.selectbox("Select Disease Type", df["Disease Type"].unique())
severity = st.selectbox("Select Severity", df["Severity"].unique())
temp = st.slider("Temperature (°C)", 20, 40, 28)
humidity = st.slider("Humidity (%)", 60, 100, 75)

# Convert input to model format
input_data = pd.DataFrame({
    "Crop Type": [encoders["Crop Type"].transform([crop])[0]],
    "Disease Type": [encoders["Disease Type"].transform([disease])[0]],
    "Severity": [encoders["Severity"].transform([severity])[0]],
    "Temperature": [temp],
    "Humidity": [humidity]
})

input_scaled = scaler.transform(input_data)

if st.button("Predict Yield Impact"):
    prediction = model.predict(input_scaled)[0]
    st.success(f"📉 Estimated Yield Impact: **{prediction:.2f}%**")

# Optional feature importance
if st.checkbox("🔍 Show Feature Importance"):
    st.subheader("Feature Importance")
    features = df.drop("Yield Impact (%)", axis=1).columns
    importances = model.feature_importances_

    fig, ax = plt.subplots()
    sns.barplot(x=importances, y=features, ax=ax)
    ax.set_title("Feature Importance")
    st.pyplot(fig)
