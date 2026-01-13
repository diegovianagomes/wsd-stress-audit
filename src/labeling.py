#%%
import pandas as pd
import numpy as np
from pathlib import Path

def get_session_start_time(subject_path):
    """Lê data absoluta do header do ACC.csv."""
    acc_file = subject_path / "ACC.csv"
    if not acc_file.exists():
        raise FileNotFoundError(f"ACC.csv não encontrado em {subject_path}")

    with open(acc_file, 'r') as f:
        date_str = f.readline().strip().split(',')[0]
    return pd.to_datetime(date_str, utc=True)

def load_tags(subject_path, session_start_dt):
    """Carrega tags e converte para segundos relativos."""
    tags_path = subject_path / "tags.csv"
    if not tags_path.exists():
        print(f"AVISO: tags.csv não encontrado em {subject_path}")
        return pd.DataFrame()

    tags = pd.read_csv(tags_path, header=None, names=['timestamp'])
    tags['dt'] = pd.to_datetime(tags['timestamp'], utc=True)
    tags['seconds_rel'] = (tags['dt'] - session_start_dt).dt.total_seconds()
    return tags

def get_stress_phases(subject_id):
    """
    Retorna os índices das fases de estresse baseado no protocolo do sujeito.
    """
    sid = subject_id.upper()
    
    # PROTOCOLO V1 (Sxx)
    # 0: Baseline
    # 1: Stroop Test (STRESS)
    # 2: Rest
    # 3: TMTC (Math) (STRESS)
    # 4: Rest
    # 5: Real Opinion (STRESS)
    # 6: Opposite Opinion (STRESS)
    # 7: Subtract Test (STRESS)
    if sid.startswith('S'):
        return [1, 3, 5, 6, 7] 
        
    # PROTOCOLO V2 (Fxx)
    # 0: Baseline
    # 1: TMTC  (STRESS)
    # 2: Rest
    # 3: Real Opinion (STRESS)
    # 4: Opposite Opinion (STRESS)
    # 5: Rest
    # 6: Subtract Test (STRESS)
    elif sid.startswith('F'):
        return [1, 3, 4, 6] 
        
    return []

def get_aerobic_phases(subject_id):
    # continue here
    return
def get_anaerobic_phases(subject_id):
    # continue here
    return
    

def apply_labels(df_sync, tags_df, subject_id):
    """Aplica labels (0=Repouso, 1=Stress) dinamicamente."""
    df_sync['label'] = 0
    if tags_df.empty: return df_sync

    stress_phases = get_stress_phases(subject_id)
    
    tag_times = tags_df['seconds_rel'].tolist()
    tag_times.append(df_sync.index.total_seconds().max())

    for i in range(len(tag_times) - 1):
        if i in stress_phases:
            start = tag_times[i]
            end = tag_times[i+1]
            
            mask = (df_sync.index.total_seconds() >= start) & \
                   (df_sync.index.total_seconds() < end)
            df_sync.loc[mask, 'label'] = 1
            
    return df_sync
# %%
