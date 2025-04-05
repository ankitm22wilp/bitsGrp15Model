import pandas as pd

def preprocess_input(df):
    df = df.copy()

    # Encode binary features if present
    if 'Is Pantry Available' in df.columns:
        df['Is Pantry Available'] = df['Is Pantry Available'].map({'Yes': 1, 'No': 0})
    else:
        df['Is Pantry Available'] = 0

    # Encode Type
    df['Type'] = df['Type'].astype('category').cat.codes if 'Type' in df.columns else 0

    # Encode Zone
    df['Zone'] = df['Zone'].astype('category').cat.codes if 'Zone' in df.columns else 0

    # Encode Classes
    if 'Classes' in df.columns:
        df['Classes'] = df['Classes'].fillna('').apply(lambda x: len(x.split(',')) if isinstance(x, str) else 0)
    else:
        df['Classes'] = 0

    # Encode Terrains
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

    # Travel Time
    if 'Arrival Time' in df.columns and 'Departure Time' in df.columns:
        arr = pd.to_datetime(df['Arrival Time'], errors='coerce')
        dep = pd.to_datetime(df['Departure Time'], errors='coerce')
        df['Travel Time'] = (arr - dep).dt.total_seconds() / 60.0
    df['Travel Time'] = df.get('Travel Time', pd.Series(0)).fillna(0)

    # Day of week
    if 'Date' in df.columns:
        df['Day_of_Week'] = pd.to_datetime(df['Date'], errors='coerce').dt.weekday
    else:
        df['Day_of_Week'] = 0

    # Departure and arrival hour/minute
    df['Departure_Hour'] = pd.to_datetime(df.get('Departure Time'), errors='coerce').dt.hour.fillna(0).astype(int)
    df['Departure_Minute'] = pd.to_datetime(df.get('Departure Time'), errors='coerce').dt.minute.fillna(0).astype(int)
    df['Arrival_Hour'] = pd.to_datetime(df.get('Arrival Time'), errors='coerce').dt.hour.fillna(0).astype(int)
    df['Arrival_Minute'] = pd.to_datetime(df.get('Arrival Time'), errors='coerce').dt.minute.fillna(0).astype(int)

    # Monthly encoding
    if 'Date' in df.columns:
        df['Month'] = pd.to_datetime(df['Date'], errors='coerce').dt.month_name()
    else:
        df['Month'] = 'January'

    for m in ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']:
        df[f'{m}_Average'] = df['Month'].apply(lambda x: 5.0 if x == m else 0.0)
        df[f'{m}_Max'] = df['Month'].apply(lambda x: 10.0 if x == m else 0.0)
        df[f'{m}_Min'] = df['Month'].apply(lambda x: 2.0 if x == m else 0.0)
        df[f'{m}_Class'] = df['Month'].apply(lambda x: 1 if x == m else 0)

    # Number of Days
    if 'Days of Run' in df.columns:
        df['Number of Days'] = df['Days of Run'].apply(lambda x: len(str(x).split(',')) if pd.notnull(x) else 0)
    else:
        df['Number of Days'] = 0

    # Drop unnecessary columns
    drop_cols = ['Train Name', 'Date', 'Departure Time', 'Arrival Time', 'Month', 'Origin', 'Destination', 'Days of Run']
    df = df.drop(columns=[col for col in drop_cols if col in df.columns])

    # Drop non-numeric columns
    for col in df.select_dtypes(include=['object']).columns:
        df = df.drop(columns=[col])

    return df

