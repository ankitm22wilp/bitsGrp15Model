import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from TrainDelayPrediction import preprocess_input

# Load trained model, label encoder, and input column order
with open("trains_delay_prediction_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("model_input_columns.pkl", "rb") as f:
    model_columns = pickle.load(f)

# Terrain options (should match what was used during training)
terrain_options = ['Coastal', 'Hills', 'Plain', 'Plains', 'Plateau']

st.title("🚆 Train Delay Category Predictor")

with st.form("delay_form"):
    st.header("Enter Train Journey Details")

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
    total_distance = st.number_input("Total Distance (in km)", min_value=1.0, max_value=5000.0, value=1000.0)
    avg_platform = st.number_input("Average Platform Count", min_value=1.0, max_value=10.0, value=3.0)
    min_platform = st.number_input("Min Platform Count", min_value=1, max_value=10, value=1)
    max_platform = st.number_input("Max Platform Count", min_value=1, max_value=10, value=5)

    # Terrain
    terrain_selected = st.multiselect("Terrain Encountered", terrain_options)

    submitted = st.form_submit_button("Predict Delay Category")

if submitted:
    try:
        # Create input dataframe
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

        # Preprocess the input
        processed = preprocess_input(input_df)

        # Reindex to match model's expected columns
        processed = processed.reindex(columns=model_columns, fill_value=0)

        # Predict
        prediction_encoded = model.predict(processed)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

        st.success(f"📊 Predicted Delay Category: **{prediction_label}**")

    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")

