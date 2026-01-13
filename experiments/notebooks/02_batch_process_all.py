# %% 1. Setup
import sys
import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import numpy as np


current_dir = Path(os.getcwd())
project_root = current_dir.parent.parent
if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DATA_DIR, PROCESSED_DIR
from src.data_loader import load_subject_raw_data
from src.preprocessing import SignalPreprocessor
from src.labeling import get_session_start_time, load_tags, apply_labels
from src.feature_extraction import FeatureExtractor

# Configurações do Artigo
EXPERIMENT = "STRESS"
WINDOW_SIZE = 60
STEP_SIZE = 30



# %% 2. Listar Sujeitos
experiment_dir = RAW_DATA_DIR / EXPERIMENT

# Lista todas as pastas dentro de STRESS/ que parecem sujeitos (f01, s01, etc)
subjects = [p.name for p in experiment_dir.iterdir() if p.is_dir()]
subjects.sort()

print(f"Encontrados {len(subjects)} sujeitos: {subjects}")

# Célula de Diagnóstico Rápido
subj = subjects[0]
subject_path = experiment_dir / subj
session_start = get_session_start_time(subject_path)
tags_df = load_tags(subject_path, session_start)

print(f"--- Diagnóstico de tags_df ({subj}) ---")
print(f"Colunas: {tags_df.columns.tolist()}")
print(tags_df.head())


# %% 3. Processamento Principal: STRESS vs REPOUSO
# Usa a lista 'subjects' e 'experiment_dir' carregadas na Célula 2

# Inicializa listas globais
all_features = []
failed_subjects = []

# Instancia Classes
extractor = FeatureExtractor(window_size=WINDOW_SIZE, step_size=STEP_SIZE)
preprocessor = SignalPreprocessor()

if 'EXCLUSIONS' not in locals():
    EXCLUSIONS = ['S02', 'f07', 'f14_a', 'f14_b', 'f17'] 

print(f"--- Processando Experimento Atual: {EXPERIMENT} ---")
print(f"Diretório: {experiment_dir}")

for subj in tqdm(subjects, desc="Stress"):
    if subj in EXCLUSIONS: continue
    
    try:
        subject_path = experiment_dir / subj
        
        # 1. Load Data
        raw_data = load_subject_raw_data(subject_path)
        if not raw_data: continue
            
        # 2. Tags & Labels (Padrão 0/1)
        session_start = get_session_start_time(subject_path)
        tags_df = load_tags(subject_path, session_start)
        df_timeline = preprocessor.synchronize_data(raw_data)
        df_labeled = apply_labels(df_timeline, tags_df, subject_id=subj)
        
        # 3. Lógica de Corte de Inércia (Específica para Stress)
        # Corta a primeira metade dos blocos de Repouso (Recuperação)
        if EXPERIMENT == "STRESS":
            df_labeled['block_id'] = (df_labeled['label'].diff() != 0).cumsum()
            indices_to_keep = []
            rest_count = 0
            
            for block, group in df_labeled.groupby('block_id'):
                lbl = group['label'].iloc[0]
                if lbl == 0: # Repouso
                    rest_count += 1
                    if rest_count > 1: # É Recuperação (corta 50%)
                        cutoff = len(group) // 2
                        indices_to_keep.extend(group.index[cutoff:])
                    else: # É Baseline (mantém tudo)
                        indices_to_keep.extend(group.index)
                else: # Stress (mantém tudo)
                    indices_to_keep.extend(group.index)
            
            df_final = df_labeled.loc[indices_to_keep].copy()
            if 'block_id' in df_final: del df_final['block_id']
        else:
            # Se por acaso não for STRESS na config, mantém como está
            df_final = df_labeled
        
        # 4. Feature Extraction
        if not df_final.empty:
            df_feats = extractor.extract_features(raw_data, df_final)
            df_feats['subject_id'] = subj
            df_feats['scenario'] = 'STRESS'
            df_feats = df_feats.dropna()
            
            if not df_feats.empty:
                all_features.append(df_feats)

    except Exception as e:
        print(f"Erro em {subj}: {e}")
        failed_subjects.append(subj)
# %% 4. Processamento ANAERÓBICO e AERÓBICO

CONFIG_EXERCISE = {
    'ANAEROBIC': {
        'label': 1, 
        'blocks': ['sprint1', 'sprint2', 'sprint3', 'sprint4']
    },
    'AEROBIC': {
        'label': 0, 
        'blocks': ['70rpm', '75rpm', '80rpm', '85rpm']
    }
}

if 'all_features' not in locals(): all_features = []

for protocol_name, config in CONFIG_EXERCISE.items():
    folder = RAW_DATA_DIR / protocol_name
    
    if not folder.exists():
        continue
        
    subjects_ex = [p.name for p in folder.iterdir() if p.is_dir()]
    subjects_ex.sort()
    
    processed_count = 0
    print(f"\n{protocol_name}")
    
    for subj in tqdm(subjects_ex, desc=f"{protocol_name}"):
        if 'EXCLUSIONS' in locals() and subj in EXCLUSIONS: continue
        
        try:
            path = folder / subj
            
            raw_data = load_subject_raw_data(path)
            if not raw_data: continue
            
            df_timeline = preprocessor.synchronize_data(raw_data)
            
            if 'time_abs' not in df_timeline.columns:
                df_timeline['time_abs'] = df_timeline.index.astype(np.int64) / 1e9

            t_start = df_timeline['time_abs'].min()
            t_end = df_timeline['time_abs'].max()
            total_duration = t_end - t_start
            
            if total_duration < 120:
                continue

            num_blocks = len(config['blocks'])
            segment_duration = total_duration / num_blocks
            
            valid_segments = []
            
            for i in range(num_blocks):
                block_start = t_start + (i * segment_duration)
                block_center = block_start + (segment_duration / 2)
                seg_start = block_center - 30
                seg_end = block_center + 30
                
                if seg_end <= t_end:
                    valid_segments.append((seg_start, seg_end, config['label']))

            if not valid_segments:
                print(f"{subj} Sem segmentos")
                continue

            dfs = []
            for t_s, t_e, lbl in valid_segments:
                mask = (df_timeline['time_abs'] >= t_s) & (df_timeline['time_abs'] <= t_e)
                seg = df_timeline.loc[mask].copy()
                seg['label'] = lbl
                dfs.append(seg)
                
            if dfs:
                df_final = pd.concat(dfs)
                df_feats = extractor.extract_features(raw_data, df_final)
                
                df_feats['subject_id'] = subj
                df_feats['scenario'] = 'EXERCISE' 
                df_feats['protocol'] = protocol_name
                df_feats = df_feats.dropna()
                
                if not df_feats.empty:
                    all_features.append(df_feats)
                    processed_count += 1

        except Exception as e:
            print(f"Erro no {subj}: {e}")
            
    print(f"{protocol_name}: {processed_count} Processado.")

#%%
main_df = pd.concat(all_features, ignore_index=True)
meta_cols = ['subject_id', 'window_id', 'label', 'scenario', 'protocol']
cols_order = [c for c in meta_cols if c in main_df.columns] + \
                [c for c in main_df.columns if c not in meta_cols]
main_df = main_df[cols_order]

# Protocolo Stress
df_stress = main_df[main_df['scenario'] == 'STRESS'].copy()

out_stress = PROCESSED_DIR / "dataset_stress.csv"
df_stress.to_csv(out_stress, index=False)

display(df_stress['label'].value_counts())

# Protocolos de Exercicio
df_exercise = main_df[main_df['scenario'] == 'EXERCISE'].copy()
out_exercise = PROCESSED_DIR / "dataset_exercise.csv"
df_exercise.to_csv(out_exercise, index=False)

display(df_exercise['label'].value_counts())

# %% -- Fim
