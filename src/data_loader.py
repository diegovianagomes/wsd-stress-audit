#%%
import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), '..')))
from src.config import SAMPLING_RATES, SENSOR_COLUMNS, RAW_DATA_DIR

#%%
def load_sensors_csv(filepath, sensor_type):
    filepath = Path(filepath)
    
    cols = SENSOR_COLUMNS.get(sensor_type)
    df = pd.read_csv(filepath, header=None, names=cols, low_memory=False)
    
    for col in df.columns:
        if sensor_type == 'IBI' and col == 'time':
            continue 
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna()
    
    if df.empty:
        return df

    if sensor_type == 'IBI':
        if 'time' in df.columns:
            df['time'] = pd.to_numeric(df['time'], errors='coerce')
            df = df.dropna()
            df = df.rename(columns={'time': 'seconds'})
            df.index = pd.to_timedelta(df['seconds'], unit='s')
            df = df.drop(columns=['seconds'])
        return df

    fs = SAMPLING_RATES.get(sensor_type)
    if not fs:
        raise ValueError(f"Frequência errada: {sensor_type}")

    df.index = pd.to_timedelta(np.arange(len(df)) / fs, unit='s')
    
    return df

#%%
def load_subject_raw_data(subject_folder_path):
    subject_folder_path = Path(subject_folder_path)
    data_bundle = {}
    
    if not subject_folder_path.exists():
        print(f"ERRO: {subject_folder_path}")
        return data_bundle

    sensors = ['ACC', 'BVP', 'EDA', 'TEMP', 'HR', 'IBI']
    
    for sensor in sensors:
        file_path = subject_folder_path / f"{sensor}.csv"
        
        if file_path.exists():
            df = load_sensors_csv(file_path, sensor)
            
            if df is not None and not df.empty:
                data_bundle[sensor] = df
            else:
                data_bundle[sensor] = pd.DataFrame()
        else:
            print(f"Vazio {file_path}")

    return data_bundle

#%%
