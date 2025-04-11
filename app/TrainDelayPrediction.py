import pandas as pd

def preprocess_input(df, skip_target_in_training=False):
    df = df.copy()

    # Encode binary features if present
    if 'Is Pantry Available' in df.columns:
        df['Is Pantry Available'] = df['Is Pantry Available'].map({'Yes': 1, 'No': 0})
    else:
        df['Is Pantry Available'] = 0

    # Encode Type safely
    df['Type'] = df['Type'].astype('category').cat.codes if 'Type' in df.columns else 0

    # Encode Zone safely
    df['Zone'] = df['Zone'].astype('category').cat.codes if 'Zone' in df.columns else 0

    # Encode Classes
    df['Classes'] = df['Classes'].fillna('').apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0) if 'Classes' in df.columns else 0

    # Parse time safely with explicit format (assuming HH:MM format)
    time_format = "%H:%M"

    if 'Departure Time' in df.columns:
        df['Departure Time Parsed'] = pd.to_datetime(df['Departure Time'], format=time_format, errors='coerce')
        df['Departure_Hour'] = df['Departure Time Parsed'].dt.hour.fillna(0).astype(int)
        df['Departure_Minute'] = df['Departure Time Parsed'].dt.minute.fillna(0).astype(int)
        df.drop(columns=['Departure Time Parsed'], inplace=True)
    else:
        df['Departure_Hour'] = 0
        df['Departure_Minute'] = 0

    if 'Arrival Time' in df.columns:
        df['Arrival Time Parsed'] = pd.to_datetime(df['Arrival Time'], format=time_format, errors='coerce')
        df['Arrival_Hour'] = df['Arrival Time Parsed'].dt.hour.fillna(0).astype(int)
        df['Arrival_Minute'] = df['Arrival Time Parsed'].dt.minute.fillna(0).astype(int)
        df.drop(columns=['Arrival Time Parsed'], inplace=True)
    else:
        df['Arrival_Hour'] = 0
        df['Arrival_Minute'] = 0

    # Calculate Travel Time (minutes)
    if 'Arrival Time' in df.columns and 'Departure Time' in df.columns:
        try:
            arrival = pd.to_datetime(df['Arrival Time'], format=time_format, errors='coerce')
            departure = pd.to_datetime(df['Departure Time'], format=time_format, errors='coerce')
            travel_time = (arrival - departure).dt.total_seconds() / 60.0
            df['Travel Time'] = travel_time.fillna(travel_time.mean())
        except:
            df['Travel Time'] = 0
    else:
        df['Travel Time'] = 0

    # Day of week
    df['Day_of_Week'] = pd.to_datetime(df['Date'], errors='coerce').dt.weekday if 'Date' in df.columns else 0

    # Month encoding
    df['Month'] = pd.to_datetime(df['Date'], errors='coerce').dt.month_name() if 'Date' in df.columns else 'January'

    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']

    for m in months:
        df[f'{m}_Average'] = df['Month'].apply(lambda x: 5.0 if x == m else 0.0)
        df[f'{m}_Max'] = df['Month'].apply(lambda x: 10.0 if x == m else 0.0)
        df[f'{m}_Min'] = df['Month'].apply(lambda x: 2.0 if x == m else 0.0)
        df[f'{m}_Class'] = df['Month'].apply(lambda x: 1 if x == m else 0)

    # Number of Days based on "Days of Run"
    df['Number of Days'] = df['Days of Run'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0) if 'Days of Run' in df.columns else 0
    
    # Encode Terrain (multi-select as comma-separated string)
    if 'Terrain' in df.columns:
        from sklearn.preprocessing import MultiLabelBinarizer
        df['Terrain'] = df['Terrain'].fillna('')
        terrain_lists = df['Terrain'].apply(lambda x: [t.strip() for t in x.split(',') if t.strip()])
        mlb = MultiLabelBinarizer()
        terrain_encoded = pd.DataFrame(mlb.fit_transform(terrain_lists), columns=mlb.classes_, index=df.index)
        df = pd.concat([df.drop(columns=['Terrain']), terrain_encoded], axis=1)
    else:
        pass  # Handle later during inference

    # Drop unwanted columns
    drop_cols = ['Train Name', 'Date', 'Departure Time', 'Arrival Time', 'Month',
                 'Origin', 'Destination', 'Days of Run']
    df.drop(columns=[col for col in drop_cols if col in df.columns], inplace=True)

    # Drop non-numeric columns if any
    for col in df.columns:
        if df[col].dtype == 'object':
            #print(f"Dropping non-numeric column: {col}")
            df.drop(columns=[col], inplace=True)

    # Compute Delay Category from Average Mode Delay
    if skip_target_in_training and 'Average Mode Delay' in df.columns:
        def categorize_delay(delay):
            if pd.isna(delay):
                return 'Unknown'
            elif delay == 0:
                return 'No Delay'
            elif delay <= 10:
                return 'Low'
            elif delay <= 30:
                return 'Medium'
            elif delay <= 60:
                return 'High'
            else:
                return 'Very High'
        
        df['Delay Category'] = df['Average Mode Delay'].apply(categorize_delay)
    else:
        df['Delay Category'] = 'Unknown'

    return df
