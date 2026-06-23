#%% 
# As quatro classes (REST, STRESS, AEROBIC, ANAEROBIC) estao em arquivos
# separados, cada um com label 0 = REST e label 1 = o estado daquele arquivo.
# Montar o problema de 4 classes exige uma decisao sobre o REST, ja que ele
# aparece em todos os arquivos:
#   1. Inspeciona todos os dataset_*.csv (shape, label, protocolo, sujeitos).
#   2. Mede a sobreposicao de janelas REST entre os arquivos, para decidir se o
#      REST e compartilhado ou distinto por sessao.
#   3. Monta provisoriamente o multiclasse com um REST canonico unico.
#   4. Avalia a viabilidade de particionamento (StratifiedGroupKFold) e indica
#      a metrica.



import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
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
    raise FileNotFoundError(f"Rnada encontrado '{marker}').")

PROJECT_ROOT = _find_project_root("src")
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PROCESSED_DIR

TARGET = "label"
GROUP = "subject_id"
WIN = "window_id"
REST_SOURCE_STATE = "STRESS"
K_LIST = [10, 8, 6, 5]


# %% DIAGNOSTICO MULTICLASSE (4 classes)"

files = sorted(Path(PROCESSED_DIR).glob("dataset_*.csv"))
print(f"\nArquivo encontrado ({len(files)}):")
scen = {}            # estado -> DataFrame
for f in files:
    d = pd.read_csv(f)
    state = f.stem.replace('dataset_', '').upper() 
    proto = d['protocol'].dropna().unique().tolist() if 'protocol' in d else []
    labels = set(d[TARGET].dropna().unique().tolist()) if TARGET in d else set()

    is_scen = (labels == {0, 1})
    print(f"\n  {f.name} | shape {d.shape} | sujeitos {d[GROUP].nunique()}")
    print(f"    label: {d[TARGET].value_counts().to_dict() if TARGET in d else 'n/a'}")
    print(f"    protocolo (unicos): {proto[:6]}{' ...' if len(proto) > 6 else ''}")
    print(f"    registrado: {is_scen}"
          + (f"  -  estado '{state}'" if is_scen else " (label nao e binario {0,1})"))
    if is_scen:
        scen[state] = d

print(f"\nEstados detectados: {list(scen.keys())}")


# %% SOBREPOSICAO DE JANELAS REST entre os arquivos (chave subject_id+window_id)

print(f"\n{'-'*90}")
print("SOBREPOSICAO DE JANELAS REST entre os arquivos (chave subject_id+window_id)")
print(f"{'-'*90}")


def rest_keys(d):
    r = d.loc[d[TARGET] == 0, [GROUP, WIN]]
    return set(map(tuple, r.values))


states = list(scen.keys())
rk = {s: rest_keys(scen[s]) for s in states}
for s in states:
    print(f"  REST em {s}: {len(rk[s])} janelas")
for i in range(len(states)):
    for j in range(i + 1, len(states)):
        a, b = states[i], states[j]
        inter = len(rk[a] & rk[b])
        union = len(rk[a] | rk[b])
        jac = inter / union if union else 0
        print(f"  {a} inter {b}: {inter} comuns | Jaccard={jac:.2f}")



# %% MONTAGEM PROVISORIA DO MULTICLASSE


if REST_SOURCE_STATE not in scen:
    REST_SOURCE_STATE = states[0]

frames = []
rest_df = scen[REST_SOURCE_STATE]
rest_df = rest_df.loc[rest_df[TARGET] == 0].copy()
rest_df['state'] = 'REST'
frames.append(rest_df)
for s, d in scen.items():
    pos = d.loc[d[TARGET] == 1].copy()
    pos['state'] = s
    frames.append(pos)

multi = pd.concat(frames, ignore_index=True)
print("\nDistribuicao de classes:")
vc = multi['state'].value_counts()
print(vc.to_string())
ratio = vc.max() / vc.min()
print(f"\nRazao maior/menor classe: {ratio:.1f} para 1 | total {len(multi)} janelas")
print(f"Sujeitos: {multi[GROUP].nunique()}")


print("\nMinimo de janelas por classe entre sujeitos que possuem a classe:")
ct = pd.crosstab(multi[GROUP], multi['state'])
for c in vc.index:
    com = ct[c][ct[c] > 0]
    print(f"  {c:10s}: sujeitos com a classe = {len(com):2d} | "
          f"min = {int(com.min())} | mediana = {int(com.median())}")


# %% VIABILIDADE (StratifiedGroupKFold): minimo de janelas da classe rara por fold

X = np.zeros((len(multi), 1))
y = multi['state'].to_numpy()
groups = multi[GROUP].to_numpy()
rare = vc.index[-1]
rows = []
for k in K_LIST:
    if k > multi[GROUP].nunique():
        continue
    try:
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
        per_fold_rare = []
        per_fold_min = []
        for _, te in cv.split(X, y, groups):
            yt = y[te]
            per_fold_rare.append(int((yt == rare).sum()))
            per_fold_min.append(int(pd.Series(yt).value_counts().min()))
        rows.append({'k': k, f'min_{rare}_por_fold': min(per_fold_rare),
                     'min_qualquer_classe_por_fold': min(per_fold_min)})
    except Exception as e:
        rows.append({'k': k, f'min_{rare}_por_fold': f'erro: {e}',
                     'min_qualquer_classe_por_fold': None})
print(pd.DataFrame(rows).to_string(index=False))
