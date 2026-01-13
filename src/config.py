import os
from pathlib import Path

WORK_SEED = 42

DATA_DIR = Path("C:/DEV/mestrado/WSD/data")
RAW_DATA_DIR = DATA_DIR / "Wearable_Dataset"

SRC_DIR = Path(__file__).parent
PROJECT_ROOT = SRC_DIR.parent

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
PROCESSED_DIR = EXPERIMENTS_DIR / "processed"
RESULTS_DIR = EXPERIMENTS_DIR / "results"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Reprodutibilidade
RANDOM_SEED = 42

# Frequências 
SAMPLING_RATES = {
    'ACC': 32.0,
    'BVP': 64.0,
    'EDA': 4.0,
    'TEMP': 4.0,
    'HR': 1.0
}
SENSOR_COLUMNS = {
    'ACC': ['x', 'y', 'z'],
    'BVP': ['bvp'],
    'EDA': ['eda'],
    'TEMP': ['temp'],
    'HR': ['hr'],
    'IBI': ['time', 'ibi']
} 

# Windowing
WINDOW_SIZE = 60
STEP_SIZE = 30

# EDA
EDA_TARGET_FREQ = 64.0
EDA_SMOOTH_SIGMA_MS = 1000 #400
EDA_SMOOTH_WINDOW = 40

# BVP / PPG
BVP_FILTER_TYPE = 'cheby2'
BVP_FILTER_ORDER = 4
BVP_FILTER_RS = 20
BVP_PASSBAND = [0.5, 8.0] #[0.5, 5.0]

# Controle de Qualidade (QC)
PEAK_DIST_MIN_SEC = 0.4
IBI_MIN_MS = 300
IBI_MAX_MS = 1200
MAX_INVALID_IBI_RATIO = 0.15