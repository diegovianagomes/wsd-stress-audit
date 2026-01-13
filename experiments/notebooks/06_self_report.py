# %% 

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.io as pio
from scipy.stats import pearsonr
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')
pio.renderers.default = "notebook"

BASE_PATH = Path(r'C:\DEV\mestrado\WSD')
DATA_DIR = BASE_PATH / 'data'
PROCESSED_DIR = BASE_PATH / 'experiments' / 'processed'

STRESS_FILE = PROCESSED_DIR / 'dataset_stress.csv'
SUBJ_V1_FILE = DATA_DIR / 'Stress_Level_v1.csv'
SUBJ_V2_FILE = DATA_DIR / 'Stress_Level_v2.csv'


def load_and_process_subjective(filepath, prefix):
    df = pd.read_csv(filepath)
    id_col = next((c for c in df.columns if 'Unnamed' in c or 'Participant' in c), None)
    df['clean_id'] = df[id_col].apply(lambda x: f"{prefix}{int(''.join(filter(str.isdigit, str(x)))):02d}")
    
    
    # Calcular Delta (Reatividade)
    nums = df.select_dtypes(include=np.number)
    base_col = 'Baseline'
    if base_col not in nums.columns:
        if not nums.empty:
            base_col = nums.columns[0]
        else:
            return None

    task_cols = [c for c in nums.columns if c != base_col and 'Subtract' not in c]
    df['Reactivity'] = df[task_cols].max(axis=1) - df[base_col]
    
    return df[['clean_id', 'Reactivity']]

df_subj_v1 = load_and_process_subjective(SUBJ_V1_FILE, 'S')
df_subj_v2 = load_and_process_subjective(SUBJ_V2_FILE, 'f')

# Concatenar se ambos foram carregados com sucesso
dfs_to_concat = [d for d in [df_subj_v1, df_subj_v2] if d is not None]

if dfs_to_concat:
    df_subjective = pd.concat(dfs_to_concat)
    print(f"Dados subjetivos prontos: {len(df_subjective)} participantes.")
else:
    df_subjective = pd.DataFrame()

df_mov = pd.DataFrame()

if STRESS_FILE.exists():
    df_ml = pd.read_csv(STRESS_FILE)
    df_ml.rename(columns={'subject_id': 'participant'}, inplace=True, errors='ignore')
    
    if 'participant' in df_ml.columns:
        df_ml = df_ml[~df_ml['participant'].isin(['f07', 'f13'])]
        acc_cols = [c for c in df_ml.columns if ('acc' in c and 'mean' in c) or c in ['x_mean', 'y_mean', 'z_mean']]
        
        if acc_cols:
            df_mov = df_ml.groupby('participant')[acc_cols].mean().mean(axis=1).reset_index(name='Movement_Intensity')
            print(f"✅ Dados de movimento: {len(df_mov)} participantes.")


#  PLOT DE CORRELAÇÃO

df_corr = pd.merge(df_subjective, df_mov, left_on='clean_id', right_on='participant')

if not df_corr.empty:
    df_corr['Group'] = df_corr['participant'].apply(
        lambda x: 'Protocolo V1 (High Load)' if x.startswith('S') else 'Protocolo V2 (Low Load)'
    )
    
    fig = px.scatter(
            df_corr, 
            x='Movement_Intensity', 
            y='Reactivity', 
            color='Group', 
            symbol='Group', 
            trendline="ols", 
            hover_name='participant',
            title='<b>Correlação: Movimento Físico vs. Estresse Percebido</b><br><sup>Quanto mais intenso o movimento, maior o estresse relatado?</sup>',
            labels={
                'Movement_Intensity': 'Intensidade Média de Movimento (Acelerômetro)', 
                'Reactivity': 'Reatividade Subjetiva (Delta Stress)'
            },
            template='plotly_white',
            height=600
        )
        
    fig.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
    fig.show()
    
    r, p = pearsonr(df_corr['Movement_Intensity'], df_corr['Reactivity'])
    print(f"   Correlação de Pearson (r): {r:.4f}")
    print(f"   P-value (p): {p:.4e}")

    print("\nEstatística por Grupo:")
    for grupo in df_corr['Group'].unique():
        df_g = df_corr[df_corr['Group'] == grupo]
        if len(df_g) > 2:
            rg, pg = pearsonr(df_g['Movement_Intensity'], df_g['Reactivity'])
            print(f"   {grupo}: r={rg:.4f}, p={pg:.4e}")
        else:
            print(f"   {grupo}: Dados insuficientes para correlação.")



# %%
