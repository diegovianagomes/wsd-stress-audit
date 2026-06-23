#%% 
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
try:
    from sklearn.model_selection import StratifiedGroupKFold
    HAS_SGK = True
except ImportError:
    HAS_SGK = False
import warnings
warnings.filterwarnings('ignore')

# CONFIGURACAO

def _find_project_root(marker="src"):
    try:
        start = Path(__file__).resolve()
    except NameError:
        start = Path.cwd().resolve()
    for parent in [start, *start.parents]:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(f"nada foi encontrado'{marker}').")

PROJECT_ROOT = _find_project_root("src")
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PROCESSED_DIR

DATA_FILE = "dataset_anaerobic.csv"

TARGET = "label"
GROUP = "subject_id"
POSITIVE = 1
K_LIST = [10, 8, 6, 5]
MIN_POS_DESEJADO = 3

# %% DESCOBERTA E CARGA


print(f"\nArquivos ({PROCESSED_DIR}):")
csvs = sorted(Path(PROCESSED_DIR).glob("*.csv"))
for c in csvs:
    print(f"  {c.name}")

path = Path(PROCESSED_DIR) / DATA_FILE
if not path.exists():
    raise FileNotFoundError(
        f"\nNao encontrei {DATA_FILE} em PROCESSED_DIR. Ajuste DATA_FILE para um "
        f"dos arquivos listados acima.")

data = pd.read_csv(path)
print(f"\nCarregado: {DATA_FILE} | shape {data.shape}")
print(f"Colunas: {list(data.columns)}")

# %%  ESTRUTURA DE ROTULOS

print(f"\nValue counts de '{TARGET}':")
print(data[TARGET].value_counts().to_string())
for col in ['scenario', 'protocol']:
    if col in data.columns:
        print(f"\nValue counts de '{col}':")
        print(data[col].value_counts().to_string())

y = (data[TARGET] == POSITIVE).astype(int)
groups = data[GROUP]
n_pos, n_neg = int(y.sum()), int((1 - y).sum())
ratio = n_neg / max(n_pos, 1)
prevalencia = n_pos / len(y)
print(f"\nPositivos (ANAEROBIC): {n_pos} | Negativos (REST): {n_neg} | "
      f"razao {ratio:.1f} para 1")
print(f"Prevalencia = {prevalencia:.4f}  (AUPRC de no-skill = {prevalencia:.4f}")

# %%  POSITIVOS POR SUJEITO

per_subj = data.assign(_pos=y).groupby(GROUP)['_pos'].agg(
    n_pos='sum', n_total='count')
per_subj['n_neg'] = per_subj['n_total'] - per_subj['n_pos']
per_subj['pct_pos'] = (per_subj['n_pos'] / per_subj['n_total'] * 100).round(1)
per_subj = per_subj.sort_values('n_pos')
print(per_subj.to_string())

zero_subj = per_subj.index[per_subj['n_pos'] == 0].tolist()
print(f"\nSujeitos sem nenhum positivo ({len(zero_subj)}): {zero_subj}")
com_pos = int((per_subj['n_pos'] > 0).sum())
print(f"Sujeitos com positivos: {com_pos} de {len(per_subj)} | "
      f"mediana de positivos entre eles: "
      f"{int(per_subj.loc[per_subj.n_pos > 0, 'n_pos'].median())}")

# %%  VIABILIDADE POR ESQUEMA DE PARTICIONAMENTO


schemes = [("GroupKFold", GroupKFold)]
if HAS_SGK:
    schemes.append(("StratifiedGroupKFold", StratifiedGroupKFold))

X = np.zeros((len(y), 1))
rows = []
detail = {}
for name, Splitter in schemes:
    for k in K_LIST:
        if k > groups.nunique():
            continue
        cv = Splitter(n_splits=k)
        pos_fold = [int(y.iloc[te].sum()) for _, te in cv.split(X, y, groups)]
        rows.append({'esquema': name, 'k': k,
                     'min_pos': min(pos_fold), 'mediana_pos': int(np.median(pos_fold)),
                     'max_pos': max(pos_fold),
                     'folds_sem_pos': int(sum(p == 0 for p in pos_fold))})
        detail[(name, k)] = pos_fold

V = pd.DataFrame(rows)
print(V.to_string(index=False))
