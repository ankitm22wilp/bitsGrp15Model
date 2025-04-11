import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from datetime import datetime
from TrainDelayPrediction import preprocess_input

# Set base directory to current working directory (important for relative paths)
BASE_DIR = os.getcwd()
st.write("📁 Current BASE_DIR:", BASE_DIR)  # Debug line to verify path resolution

# Load shared label encoder
with open(os.path.join(BASE_DIR, "label_encoder.pkl"), "rb") as f:
    label_encoder = pickle.load(f)

# Define model-specific folders
model_folders = {
    "Random Forest": "random_forest",
    "XGBoost": "xgboost",
    "Logistic Regression": "logistic_regression",
    "SVM": "svm"
}

# Terrain options
terrain_options = ['Coastal', 'Hills', 'Plain', 'Plains', 'Plateau']

st.title("🚆 Train Delay Category Predictor")

with st.form("delay_form"):
    st.header("Enter Train Journey Details")

    # Model selection
    selected_model_name = st.selectbox("Select Prediction Model", list(model_folders.keys()))
    model_dir = os.path.join(BASE_DIR, model_folders[selected_model_name])

    # Basic Inputs
    train_type = st.selectbox("Train Type", ["Super Fast", "Express", "Rajdhani", "Duronto", "Mail"])
    zone = st.selectbox("Railway Zone", ["SWR", "NR", "CR", "ER", "WR", "NWR", "ECR", "SCR"])
    coach_count = st.number_input("Number of Coaches", min_value=5, max_value=30, value=20)
    pantry = st.selectbox("Is Pantry Available?", ["Yes", "No"])
    days_of_run = st.multiselect("Days of Run", ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'])
    travel_date = st.date_input("Travel Date", value=datetime.today().date())

    # Time Inputs
    dep_time = st.time_input("Departure Time", value=datetime.strptime("12:00", "%H:%M").time())
    arr_time = st.time_input("Arrival Time", value=datetime.strptime("18:00", "%H:%M").time())

    # Classes
    classes_input = st.multiselect("Available Classes", ['1A', '2A', '3A', 'SL', 'CC', '3E'])

    # Station-level inputs
    num_stations = st.number_input("Number of Stations", min_value=1, max_value=150, value=10)
    total_distance = st.number_input("Total Distance (in km)", min_value=1.0, max_value=5000.0, value=1500.0)
    avg_platform = st.number_input("Average Platform Count", min_value=1, max_value=10, value=5)
    min_platform = st.number_input("Min Platform Count", min_value=1, max_value=10, value=1)
    max_platform = st.number_input("Max Platform Count", min_value=1, max_value=10, value=5)

    # Terrain
    terrain_selected = st.multiselect("Terrain Encountered", terrain_options)

    submitted = st.form_submit_button("Predict Delay Category")

if submitted:
    try:
        # Build input row
        input_data = {
            'Type': train_type,
            'Zone': zone,
            'Coach Count': coach_count,
            'Is Pantry Available': pantry,
            'Departure Time': dep_time.strftime("%H:%M"),
            'Arrival Time': arr_time.strftime("%H:%M"),
            'Date': pd.to_datetime(travel_date).strftime("%Y-%m-%d"),
            'Days of Run': ",".join(days_of_run),
            'Classes': ",".join(classes_input),
            'Num_Stations': num_stations,
            'Total_Distance': total_distance,
            'Avg_Platform_Count': avg_platform,
            'Min_Platform_Count': min_platform,
            'Max_Platform_Count': max_platform,
            'Terrain': ",".join(terrain_selected)
        }

        input_df = pd.DataFrame([input_data])
        processed = preprocess_input(input_df)

        # Load model input columns
        with open(os.path.join(model_dir, "model_input_columns.pkl"), "rb") as f:
            model_columns = pickle.load(f)

        # Reindex to match model input format
        processed = processed.reindex(columns=model_columns, fill_value=0)

        # Load model
        with open(os.path.join(model_dir, "model.pkl"), "rb") as f:
            model = pickle.load(f)

        # For models that need imputation
        if selected_model_name in ["Logistic Regression", "SVM"]:
            with open(os.path.join(model_dir, "imputer.pkl"), "rb") as f:
                imputer = pickle.load(f)
            processed = imputer.transform(processed)

        # Predict
        prediction = model.predict(processed)[0]

        # Small fix to support both encoded and string-based model outputs
        prediction_label = prediction if isinstance(prediction, str) else label_encoder.inverse_transform([prediction])[0]

        st.success(f"📊 Predicted Delay Category using **{selected_model_name}**: **{prediction_label}**")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

