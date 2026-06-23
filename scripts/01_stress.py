"""
Pipeline de extração de features — Cenário 1 (REST vs STRESS)

Processa os sinais fisiológicos dos protocolos de exercício do artigo/dataset, 
cada protocolo gera um CSV separado com features das fases:

Dados brutos (CSV por sensor)
    - Sincronização temporal (4Hz comum)
    - Rotulagem por protocolo (tags.csv - fases REST/STRESS)
    - Janelamento (60s janela, 30s passo, sobreposição 50%)
    - Extração de 49 features (EDA, HRV, ACC, BVP, HR)
    - CSV final com uma linha por janela

Dataset: WSD (Hongn et al., 2025) com 36 sujeitos e usando o Empatica E4

Grupos: 
    V1 (Sxx, 8 fases) 
    V2 (Fxx, 7 fases)
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


# %%
"""
Define qual experimento processar e quais sujeitos excluir.

Exclusões (problemas nos dados brutos do WSD):
- S02         -> arquivo corrompido / incompleto
- f07         -> sessão interrompida
- f14_a ou _b -> sujeito duplicado (duas tentativas)
- f17         -> dados faltantes

O script lista automaticamente as pastas de sujeitos dentro do diretório
do experimento. Cada pasta contém os CSVs por sensor
(ACC.csv, BVP.csv, EDA.csv, TEMP.csv, HR.csv, IBI.csv, tags.csv).
"""

EXPERIMENT = "STRESS"
EXPERIMENT_DIR = RAW_DATA_DIR / EXPERIMENT
EXCLUSIONS = ['S02', 'f07', 'f13', 'f14_a', 'f14_b', 'f17']
subjects = sorted([p.name for p in EXPERIMENT_DIR.iterdir() if p.is_dir()])
subjects_valid = [s for s in subjects if s not in EXCLUSIONS]

print(f"Total: {len(subjects)}\nVálidos: {len(subjects_valid)}\nExcluídos: {EXCLUSIONS}")
print(f"Sujeitos: {subjects_valid}")

# %% Função de corte de inércia
"""
Reimplementação do protocolo original (Hongn et al., 2025) que utilizou a segunda metade de cada um 
dos dois blocos de descanso para minimizar a influencia de qualquer efeito de estresse residual.

Baseline que é o primeiro bloco de repouso, é mantido integralmente e Blocos de repouso subsequentes com recuperação entre tarefas têm a 
primeira metade removida.
"""

def apply_inertia_cut(df_labeled):
    """Remove a primeira metade dos blocos de repouso pós-baseline."""
    df = df_labeled.copy()
    df['block_id'] = (df['label'].diff() != 0).cumsum()
    
    indices_to_keep = []
    rest_count = 0

    for block_id, group in df.groupby('block_id', sort=False):
        lbl = group['label'].iloc[0]
        
        if lbl == 0:  # Repouso
            rest_count += 1
            if rest_count > 1:  # Recuperação e não baseline
                cutoff = len(group) // 2
                indices_to_keep.extend(group.index[cutoff:])
            else:  # Baseline: mantém tudo
                indices_to_keep.extend(group.index)
        else:  # Stress: mantém tudo
            indices_to_keep.extend(group.index)
    
    df_cut = df.loc[indices_to_keep].copy()
    df_cut = df_cut.drop(columns=['block_id'])
    return df_cut

# %%
"""
Processa um único sujeito para validar o pipeline antes do batch.
Serve como debug rápido que se acontecer algum erro aqui, não vale rodar os 32.

O fluxo de cada sujeito:
- load_subject_raw_data  - lê os 6 CSVs e retorna dict {sensor: DataFrame}
- get_session_start_time - lê timestamp absoluto do header do ACC.csv
- load_tags              - converte timestamps das tags para segundos relativos
- synchronize_data       - alinha todos os sensores para 4Hz (índice comum)
- apply_labels           - marca label 0 (REST) ou 1 (STRESS) com base nas 
                           fases do protocolo (V1 ou V2, detectado pelo ID)
- extract_features       - janela deslizante sobre os dados brutos, usando
                           df_labeled apenas para atribuir o label da janela
                           (moda dos labels dentro da janela)
- dropna                 - remove janelas de borda com dados insuficientes
"""

TEST_SUBJECT = "f01"
subject_path = EXPERIMENT_DIR / TEST_SUBJECT
raw_data = load_subject_raw_data(subject_path)
print(f"Sensores carregados: {list(raw_data.keys())}")

session_start = get_session_start_time(subject_path)
tags_df = load_tags(subject_path, session_start)
print(f"Tags: {len(tags_df)} \nInício sessão: {session_start}")

preprocessor = SignalPreprocessor()
df_timeline = preprocessor.synchronize_data(raw_data)
df_labeled = apply_labels(df_timeline, tags_df, subject_id=TEST_SUBJECT, protocol='STRESS')

counts = df_labeled['label'].value_counts()
print(f"\nRepouso: {counts.get(0, 0)} amostras \n Stress: {counts.get(1, 0)} amostras")

extractor = FeatureExtractor(window_size=WINDOW_SIZE, step_size=STEP_SIZE)
df_features = extractor.extract_features(raw_data, df_labeled)
df_features = df_features.dropna()

# Corte de inércia
print(f"Corte de Inércia, antes e depois do Corte\n")
print(f"Antes: {len(df_labeled)} amostras\n - Repouso(REST) = {sum(df_labeled['label']==0)}\n - Estresse(STRESS) = {sum(df_labeled['label']==1)}")
df_labeled = apply_inertia_cut(df_labeled)
print(f"Depois: {len(df_labeled)} amostras\n - Repouso(REST) = {sum(df_labeled['label']==0)}\n - Estresse(STRESS) = {sum(df_labeled['label']==1)}")

print(f"Features extraídas: {df_features.shape}")
print(f"Colunas: {list(df_features.columns)}")

# %%
"""
Verifica visualmente se a extração faz sentido antes de rodar o batch.

Plota a média da atividade eletrodermica(EDA) bruta ao longo das janelas, colorido pelo label e é esperado ver valores mais altos (vermelho) nas fases de estresse e 
mais baixos (azul) no repouso.

Se o gráfico mostrar cores misturadas sem padrão, algo deu errado na rotulagem ou na sincronização.
"""

fig, axes = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

axes[0].scatter(df_features['window_id'], df_features['mean_raw_eda'],
                c=df_features['label'], cmap='coolwarm', alpha=0.6, s=20)
axes[0].set_ylabel("Média EDA (µS)")
axes[0].set_title(f"Sanidade: {TEST_SUBJECT} | {df_features.shape[0]} janelas")

axes[1].scatter(df_features['window_id'], df_features['hr_mean'],
                c=df_features['label'], cmap='coolwarm', alpha=0.6, s=20)
axes[1].set_ylabel("Média HR (bpm)")
axes[1].set_xlabel("Tempo da janela em segundos")

plt.tight_layout()
plt.show()

print(f"Labels no dataset de features:")
print(df_features['label'].value_counts().to_string())
#%%
"""
Verifica se as features ['total_power', 'LF_power', 'HF_power', 'ratio'] foram processadas, elas são importantes para o experimento
"""
print(df_features[['total_power', 'LF_power', 'HF_power', 'ratio']].describe())

# %% 
"""
Roda o mesmo pipeline da celula anterior para todos os sujeitos válidos.

Para cada sujeito:
    - Carrega dados brutos
    - Sincroniza, rotula, extrai features
    - Adiciona coluna 'subject_id' para identificar a origem

Sujeitos que falharem são registrados em 'failed_subjects' para inspeção 
posterior, sem interromper o loop.

Saída: lista de DataFrames (um por sujeito), prontos para concatenação.
"""

all_features = []
failed_subjects = []

extractor = FeatureExtractor(window_size=WINDOW_SIZE, step_size=STEP_SIZE)
preprocessor = SignalPreprocessor()

for subj in tqdm(subjects_valid, desc="Processando STRESS"):
    try:
        subject_path = EXPERIMENT_DIR / subj

        raw_data = load_subject_raw_data(subject_path)
        if not raw_data:
            failed_subjects.append((subj, "Dados vazios"))
            continue

        session_start = get_session_start_time(subject_path)
        tags_df = load_tags(subject_path, session_start)
        df_timeline = preprocessor.synchronize_data(raw_data)
        df_labeled = apply_labels(df_timeline, tags_df, subject_id=subj, protocol='STRESS')
        df_labeled = apply_inertia_cut(df_labeled)


        # Extrai features
        df_feats = extractor.extract_features(raw_data, df_labeled)
        df_feats['subject_id'] = subj
        df_feats = df_feats.dropna()

        if not df_feats.empty:
            all_features.append(df_feats)
            print(f"  {subj}: {df_feats.shape[0]} janelas | REST={sum(df_feats['label']==0)} STRESS={sum(df_feats['label']==1)}")

    except Exception as e:
        print(f"  ERRO {subj}: {e}")
        failed_subjects.append((subj, str(e)))

print(f"\nProcessados: {len(all_features)} | Falhas: {len(failed_subjects)}")
if failed_subjects:
    print("Falhas:", failed_subjects)

#%% 
"""
Concatena os DataFrames de todos os sujeitos em um dataset único.

Organiza as colunas com metadados (subject_id, label) à esquerda e 
features à direita. Salva como CSV em PROCESSED_DIR.

Este CSV é a entrada direta para o script de modelagem.
"""

df_stress = pd.concat(all_features, ignore_index=True)

meta_cols = ['subject_id', 'window_id', 'label']
feature_cols = [c for c in df_stress.columns if c not in meta_cols]
df_stress = df_stress[meta_cols + feature_cols]

print(f"Dataset final: {df_stress.shape}")
print(f"Sujeitos: {df_stress['subject_id'].nunique()}")
print(f"\nDistribuição de labels:")
print(df_stress['label'].value_counts().to_string())
print(f"\nJanelas por sujeito:")
print(df_stress.groupby('subject_id')['label'].count().to_string())

output_path = PROCESSED_DIR / "dataset_stress.csv"
df_stress.to_csv(output_path, index=False)
print(f"\nSalvo em: {output_path}")

# %% Fim

