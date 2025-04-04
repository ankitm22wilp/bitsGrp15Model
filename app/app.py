import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model = pickle.load(open("trains_delay_prediction_model.pkl", "rb"))

st.set_page_config(page_title="Train Delay Predictor", layout="centered")
st.title("🚉 Train Delay Prediction")
st.write("Predict whether a train is likely to face **Low**, **Moderate**, or **High** delays based on its features.")

# --- UI Inputs ---
distance = st.slider("Distance (in km)", 0, 3000, 1000)

# Select Type
train_type = st.selectbox("Train Type", ['Express', 'Superfast', 'Passenger'])

# Select Zone
zone = st.selectbox("Zone", ['NR', 'WR', 'CR', 'ER', 'SR', 'ECR', 'WCR'])  # Add your real zones

# Select Month
month = st.selectbox("Month", ['Jan', 'Feb', 'March', 'April', 'May', 'June',
                               'July', 'August', 'September', 'October', 'November', 'December'])

# Select Classes (simplified version)
classes = st.multiselect("Available Classes", ['1A', '2A', '3A', 'SL', 'CC'])

# --- Prepare Input Vector ---
# Create a dictionary to simulate a row of input data
input_dict = {
    'Distance': distance,
    'Type_Express': int(train_type == 'Express'),
    'Type_Passenger': int(train_type == 'Passenger'),
    'Type_Superfast': int(train_type == 'Superfast'),
    'Zone_' + zone: 1,
}

# Add class columns
for cls in ['1A', '2A', '3A', 'SL', 'CC']:
    input_dict[f'Has_{cls}'] = int(cls in classes)

# Add month one-hot
for m in ['Jan', 'Feb', 'March', 'April', 'May', 'June',
          'July', 'August', 'September', 'October', 'November', 'December']:
    input_dict[f'Month_{m}'] = int(m == month)

# Convert to DataFrame with all missing columns filled with 0
X_input = pd.DataFrame([input_dict])
model_features = model.feature_names_in_ if hasattr(model, 'feature_names_in_') else X_input.columns
X_input = X_input.reindex(columns=model_features, fill_value=0)

# --- Predict ---
if st.button("Predict Delay Category"):
    pred = model.predict(X_input)[0]
    label_map = {0: 'Low Delay 🚦', 1: 'Moderate Delay 🕒', 2: 'High Delay ⛔️'}
    st.success(f"### Prediction: **{label_map.get(pred, 'Unknown')}**")
