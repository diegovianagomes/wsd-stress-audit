#%%
import pandas as pd
import numpy as np
from pathlib import Path

#%%[Markdown]
# Lê data absoluta do header do ACC.csv.
def get_session_start_time(subject_path):
    acc_file = subject_path / "ACC.csv"
    if not acc_file.exists():
        raise FileNotFoundError(f"ACC.csv não encontrado em {subject_path}")

    with open(acc_file, 'r') as f:
        date_str = f.readline().strip().split(',')[0]
    return pd.to_datetime(date_str, utc=True)

#%%[Markdown]
# Carrega tags e converte para segundos relativos.
def load_tags(subject_path, session_start_dt):

    tags_path = subject_path / "tags.csv"
    if not tags_path.exists():
        return pd.DataFrame()

    tags = pd.read_csv(tags_path, header=None, names=['timestamp'])
    tags['dt'] = pd.to_datetime(tags['timestamp'], utc=True)
    tags['seconds_rel'] = (tags['dt'] - session_start_dt).dt.total_seconds()
    return tags

#%%[Markdown]
# Retorna os índices das fases, baseado no protocolo de cada sujeito.
# PROTOCOLO V1 (Sxx)
    # 0: Baseline
    # 1: Stroop Test (STRESS)
    # 2: Rest
    # 3: TMTC (Math) (STRESS)
    # 4: Rest
    # 5: Real Opinion (STRESS)
    # 6: Opposite Opinion (STRESS)
    # 7: Subtract Test (STRESS)

 # PROTOCOLO V2 (Fxx)
    # 0: Baseline
    # 1: TMTC  (STRESS)
    # 2: Rest
    # 3: Real Opinion (STRESS)
    # 4: Opposite Opinion (STRESS)
    # 5: Rest
    # 6: Subtract Test (STRESS)

def get_stress_phases(subject_id):

    sid = subject_id.upper()
    
    if sid.startswith('S'):
        return [1, 3, 5, 6, 7] 
        

    elif sid.startswith('F'):
        return [1, 3, 4, 6] 
        
    return []

def get_anaerobic_phases(subject_id):
    """
    Fases de sprint no apply_labels (conta a partir do primeiro tag).
    V1 (Sxx, 7 tags): Sprint1=fase0, Sprint2=fase2, Sprint3=fase4
    V2 (Fxx, 11 tags): Sprint1=fase1, Sprint2=fase3, Sprint3=fase5, Sprint4=fase7
    """
    sid = subject_id.upper()
    if sid.startswith('S'):
        return [0, 2, 4]
    elif sid.startswith('F'):
        return [1, 3, 5, 7]
    return []

def get_aerobic_phases(subject_id):
    """
    Fases de ciclismo no apply_labels (conta a partir do primeiro tag).
    V1 (Sxx, 12 tags): 10 estágios (60-110 rpm) = fases 0..9
    V2 (Fxx, 8 tags): 4 estágios (70, 75, 85, 90/95) = fases 1..4
    """
    sid = subject_id.upper()
    if sid.startswith('S'):
        return list(range(0, 10))
    elif sid.startswith('F'):
        return [1, 2, 3, 4]
    return []
#%%[Markdown]
# Aplica labels (0=Repouso, 1=Stress)
def apply_labels(df_sync, tags_df, subject_id, protocol='STRESS'):
    """
    Aplica labels binários (0=repouso/não-exercício, 1=estresse/exercício).
    
    O mapeamento de fases depende do protocolo e da versão (Sxx vs Fxx):
        STRESS:    get_stress_phases
        ANAEROBIC: get_anaerobic_phases
        AEROBIC:   get_aerobic_phases
    """
    df_sync['label'] = 0
    if tags_df.empty:
        return df_sync

    # Seleciona as fases ativas conforme o protocolo
    phase_map = {
        'STRESS': get_stress_phases,
        'ANAEROBIC': get_anaerobic_phases,
        'AEROBIC': get_aerobic_phases,
    }
    
    get_phases = phase_map.get(protocol)
    if not get_phases:
        raise ValueError(f"Protocolo desconhecido: {protocol}")
    
    active_phases = get_phases(subject_id)
    
    tag_times = tags_df['seconds_rel'].tolist()
    tag_times.append(df_sync.index.total_seconds().max())

    for i in range(len(tag_times) - 1):
        if i in active_phases:
            start = tag_times[i]
            end = tag_times[i + 1]
            mask = (df_sync.index.total_seconds() >= start) & \
                   (df_sync.index.total_seconds() < end)
            df_sync.loc[mask, 'label'] = 1

    return df_sync
# %%
