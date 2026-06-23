#%%[Markdown]
"""
Pipeline de extração de features dos Protocolos de Exercício, AEROBIC e ANAEROBIC.

Processa os sinais fisiológicos dos protocolos de exercício do artigo/dataset, cada protocolo gera um CSV separado com features das fases:

 - Exercício:                 (label=1)
 - Repouso/baseline/cooldown: (label=0)

Protocolos:
    - ANAEROBIC V1 (Sxx): 3 sprints de 30s (Wingate adaptado)
    - ANAEROBIC V2 (Fxx): 4 sprints de 45s
    - AEROBIC V1 (Sxx)  : 10 estágios de ciclismo (60-110 rpm, Storer-Davis adaptado)
    - AEROBIC V2 (Fxx)  : 4 estágios de ciclismo (70-95 rpm, simplificado)

Para o Cenário 2 (AEROBIC vs ANAEROBIC), a combinação dos datasets é feita
no script de modelagem, usando apenas as janelas de exercício (label=1).
"""

#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import warnings
warnings.filterwarnings('ignore', message='Mean of empty slice')

from src.config import RAW_DATA_DIR, PROCESSED_DIR, WINDOW_SIZE, STEP_SIZE
from src.data_loader import load_subject_raw_data
from src.preprocessing import SignalPreprocessor
from src.labeling import get_session_start_time, load_tags, apply_labels
from src.feature_extraction import FeatureExtractor


# %% 2. Configuração
'''
As exclusões são baseadas no 'data_constraints.txt' do artigo/dataset:

AEROBIC:
    S03 -> encerrou protocolo em 90 rpm (incompleto)
    S07 -> encerrou protocolo em 95 rpm (incompleto)
    S11 -> conexão Bluetooth perdida (arquivos split: S11_a, S11_b)
    S12 -> não realizou protocolo aeróbico (sem dados)

ANAEROBIC:
    S06 -> encerrou protocolo antes (último sprint faltando)
    S16 -> conexão Bluetooth perdida (arquivos split: S16_a, S16_b)
'''

PROTOCOLS = ['ANAEROBIC', 'AEROBIC']

EXCLUSIONS = {
    'ANAEROBIC': ['S06', 'S16', 'S16_a', 'S16_b'],
    'AEROBIC':   ['S03', 'S07', 'S11', 'S11_a', 'S11_b', 'S12'],
}

for protocol in PROTOCOLS:
    protocol_dir = RAW_DATA_DIR / protocol
    if not protocol_dir.exists():
        print(f"AVISO: {protocol_dir} não encontrado")
        continue
    
    subjects = sorted([p.name for p in protocol_dir.iterdir() if p.is_dir()])
    valid = [s for s in subjects if s not in EXCLUSIONS.get(protocol, [])]
    print(f"{protocol}: {len(subjects)} total | {len(valid)} válidos | Excluídos: {EXCLUSIONS[protocol]}")
    print(f"  Sujeitos: {valid}")

# %% 3. Teste com um sujeito (sanidade)
"""
Processa um sujeito de cada protocolo para validar o pipeline e ussa 'apply_labels' como protocolo'ANAEROBIC' ou 'AEROBIC' para selecionar
as fases corretas automaticamente (V1/V2 detectado pelo ID).
"""

preprocessor = SignalPreprocessor()
extractor = FeatureExtractor(window_size=WINDOW_SIZE, step_size=STEP_SIZE)

for protocol in PROTOCOLS:
    protocol_dir = RAW_DATA_DIR / protocol
    if not protocol_dir.exists(): continue
    
    # Pega primeiro sujeito válido
    subjects = sorted([p.name for p in protocol_dir.iterdir() if p.is_dir()])
    test_subj = [s for s in subjects if s not in EXCLUSIONS.get(protocol, [])][0]
    subject_path = protocol_dir / test_subj
    
    print(f"=== {protocol} / {test_subj} ===")
    
    raw_data = load_subject_raw_data(subject_path)
    session_start = get_session_start_time(subject_path)
    tags_df = load_tags(subject_path, session_start)
    print(f"Sensores: {list(raw_data.keys())} | Tags: {len(tags_df)}")
    
    df_timeline = preprocessor.synchronize_data(raw_data)
    df_labeled = apply_labels(df_timeline, tags_df, subject_id=test_subj, protocol=protocol)
    
    counts = df_labeled['label'].value_counts()
    print(f"Repouso: {counts.get(0, 0)} | Exercício: {counts.get(1, 0)}")
    
    df_features = extractor.extract_features(raw_data, df_labeled)
    df_features = df_features.dropna()
    print(f"Features: {df_features.shape}")
    print(f"Labels: {df_features['label'].value_counts().to_dict()}")

# %% 4. Processamento batch — ANAEROBIC e AEROBIC
"""
Processa todos os sujeitos válidos de cada protocolo e vai gerar um DataFrame por protocolo, com colunas 'subject_id' e 'protocol'.
"""

preprocessor = SignalPreprocessor()
extractor = FeatureExtractor(window_size=WINDOW_SIZE, step_size=STEP_SIZE)

results = {}

for protocol in PROTOCOLS:
    protocol_dir = RAW_DATA_DIR / protocol
    if not protocol_dir.exists(): continue
    
    subjects = sorted([p.name for p in protocol_dir.iterdir() if p.is_dir()])
    valid = [s for s in subjects if s not in EXCLUSIONS.get(protocol, [])]
    
    all_features = []
    failed = []
    
    for subj in tqdm(valid, desc=protocol):
        try:
            subject_path = protocol_dir / subj
            
            raw_data = load_subject_raw_data(subject_path)
            if not raw_data:
                failed.append((subj, "Dados vazios"))
                continue
            
            session_start = get_session_start_time(subject_path)
            tags_df = load_tags(subject_path, session_start)
            df_timeline = preprocessor.synchronize_data(raw_data)
            df_labeled = apply_labels(df_timeline, tags_df, subject_id=subj, protocol=protocol)
            
            df_feats = extractor.extract_features(raw_data, df_labeled)
            df_feats['subject_id'] = subj
            df_feats['protocol'] = protocol
            df_feats = df_feats.dropna()
            
            if not df_feats.empty:
                all_features.append(df_feats)
                ex = sum(df_feats['label'] == 1)
                rest = sum(df_feats['label'] == 0)
                print(f"  {subj}: {df_feats.shape[0]} janelas para Repouso = {rest} Exercicio = {ex}")
        
        except Exception as e:
            print(f"  ERRO {subj}: {e}")
            failed.append((subj, str(e)))
    
    results[protocol] = all_features
    print(f"\n{protocol}: {len(all_features)} sujeitos e falhas: {len(failed)}")
    if failed: print(f"  falhas: {failed}")

# %% 5. Consolidação e salvamento
"""
Salva um CSV por protocolo (dataset_anaerobic.csv, dataset_aerobic.csv) e esses CSVs são entrada para o Cenário 2 (AEROBIC vs ANAEROBIC) no script
de modelagem, onde as janelas de exercício (label=1) de ambos são combinadas com rótulos distintos.
"""

meta_cols = ['subject_id', 'window_id', 'label', 'protocol']

for protocol in PROTOCOLS:
    if protocol not in results or not results[protocol]:
        print(f"{protocol}: sem dados")
        continue
    
    df = pd.concat(results[protocol], ignore_index=True)
    feature_cols = [c for c in df.columns if c not in meta_cols]
    df = df[[c for c in meta_cols if c in df.columns] + feature_cols]
    
    output_path = PROCESSED_DIR / f"dataset_{protocol.lower()}.csv"
    df.to_csv(output_path, index=False)
    
    print(f"{protocol}")
    print(f"Shape: {df.shape}")
    print(f"Sujeitos: {df['subject_id'].nunique()}")
    print(f"Labels: {df['label'].value_counts().to_dict()}")
    print(f"Salvo: {output_path}")

# %%
