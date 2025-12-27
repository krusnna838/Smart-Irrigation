import pandas as pd
import numpy as np
import os
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# File paths
DATA_FILE = "irrigation_data.csv"
HISTORY_FILE = "prediction_history.csv"

# --- AUTOMATIC CSV INITIALIZATION ---
def initialize_files():
    # 1. Create Training Data if missing (for the AI model)
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame({
            "soil_moisture": np.random.randint(10, 80, 100),
            "temperature": np.random.randint(15, 45, 100),
            "humidity": np.random.randint(30, 90, 100),
            "rainfall": np.random.randint(0, 20, 100),
            "irrigation_needed": np.random.randint(5, 60, 100)
        })
        df.to_csv(DATA_FILE, index=False)

    # 2. Create History CSV if missing (for the System page)
    if not os.path.exists(HISTORY_FILE):
        headers = ["Timestamp", "Crop", "Soil_Moisture", "Temperature", "Humidity", "Rainfall", "Result"]
        pd.DataFrame(columns=headers).to_csv(HISTORY_FILE, index=False)

initialize_files()

# --- MODEL TRAINING ---
df = pd.read_csv(DATA_FILE)
X = df.drop("irrigation_needed", axis=1)
y = df["irrigation_needed"]
model = RandomForestRegressor(n_estimators=100).fit(X, y)

def save_to_history(crop, soil, temp, hum, rain, result):
    """Logs prediction data to the CSV automatically"""
    new_entry = {
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Crop": crop,
        "Soil_Moisture": soil,
        "Temperature": temp,
        "Humidity": hum,
        "Rainfall": rain,
        "Result": round(result, 2)
    }
    df_history = pd.DataFrame([new_entry])
    df_history.to_csv(HISTORY_FILE, mode='a', index=False, header=False)

def predict_irrigation(soil, temp, hum, rain):
    input_df = pd.DataFrame([[soil, temp, hum, rain]], 
                            columns=["soil_moisture", "temperature", "humidity", "rainfall"])
    return model.predict(input_df)[0]