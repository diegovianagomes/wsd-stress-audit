#%% 

#   1. Computa a disparidade por PREFIXO (f vs S) a partir dos subject_report,
#      como sinal preliminar (rotulado como prefixo/coorte, nao sexo).
#   2. Procura, no projeto, um arquivo de demografia com sexo/idade.
#   3. Se achar sexo, cruza prefixo contra sexo para revelar o confundimento, e
#      so entao agrupa por sexo de verdade.

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
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
    raise FileNotFoundError(f"nenhum arquivo  '{marker}').")

PROJECT_ROOT = _find_project_root("src")

MODELS = ['RF', 'XGB', 'LGBM']
TARGET_SUBJECTS = ['f02', 'f03', 'f04', 'f18'] 

DEMO_PATH = None
SEX_KEYS = ['sex', 'gender', 'sexo', 'genero']
AGE_KEYS = ['age', 'idade']
SUBJ_KEYS = ['subject_id', 'subject', 'info', 'id', 'participant', 'participante']


# %% CARREGA subject_report E COMPUTA POR SUJEITO
reps = {}
for m in MODELS:
    hits = list(PROJECT_ROOT.rglob(f"subject_report_{m}.csv"))
    if hits:
        reps[m] = pd.read_csv(hits[0]).set_index('subject')
        print(f"  subject_report_{m}.csv -> {hits[0]}")
    else:
        print(f"ERRO")

if not reps:
    raise FileNotFoundError("nada encontrado")

auc = pd.concat([reps[m]['auc'] for m in reps], axis=1).mean(axis=1).rename('auc')
rec = pd.concat([reps[m]['recall_stress'] for m in reps], axis=1).mean(
    axis=1).rename('recall')
S = pd.concat([auc, rec], axis=1)
S['fnr'] = 1 - S['recall']               # taxa de falsos negativos da classe estresse
S['prefix'] = [str(i)[0] for i in S.index]
print(f"\nSujeitos com metricas: {len(S)} | prefixos: "
      f"{S['prefix'].value_counts().to_dict()}")


# %% DISPARIDADE POR PREFIXO f vs S  (preliminar; confundido com protocolo)

grp = S.groupby('prefix').agg(
    n=('auc', 'size'), auc_mean=('auc', 'mean'), auc_median=('auc', 'median'),
    recall_mean=('recall', 'mean'), fnr_mean=('fnr', 'mean'))
print(grp.round(3).to_string())

if set(S['prefix'].unique()) >= {'f', 'S'}:
    a = S.loc[S['prefix'] == 'f', 'auc']
    b = S.loc[S['prefix'] == 'S', 'auc']
    u, p = mannwhitneyu(a, b, alternative='two-sided')
    print(f"\nMann-Whitney AUC (f x S): U={u:.1f}  p={p:.4f}  "
          f"-> {'diferenca significativa' if p < 0.05 else 'sem diferenca significativa'}")
    print(f"  AUC mediano: f={a.median():.3f} | S={b.median():.3f} "
          f"(n_f={len(a)}, n_S={len(b)}, poder baixo)")

print(S.loc[[s for s in TARGET_SUBJECTS if s in S.index],
            ['auc', 'recall', 'fnr', 'prefix']].round(3).to_string())


# %% BUSCA POR ARQUIVO DE DEMOGRAFIA (sexo/idade)


def _has_key(cols, keys):
    low = [c.lower() for c in cols]
    return any(any(k == c or k in c for c in low) for k in keys)


candidates = []
for csv in PROJECT_ROOT.rglob("*.csv"):
    try:
        cols = pd.read_csv(csv, nrows=0).columns.tolist()
    except Exception:
        continue
    if _has_key(cols, SEX_KEYS):
        candidates.append((csv, cols))

if DEMO_PATH is None and candidates:
    for c, cols in candidates:
        print(f"  {c}  | colunas: {cols}")
    DEMO_PATH = candidates[0][0]
elif DEMO_PATH is None:



# %% PCRUZA PREFIXO x SEXO E AUDITA POR SEXO 

if DEMO_PATH is not None:
    demo = pd.read_csv(DEMO_PATH)
    demo.columns = demo.columns.str.strip() 
    cols_low = {c.lower(): c for c in demo.columns}
    subj_col = next((cols_low[k] for k in SUBJ_KEYS if k in cols_low), None)
    if subj_col is None:
        subj_col = demo.columns[0]
    sex_col = next((cols_low[k] for k in SEX_KEYS if k in cols_low), None)
    age_col = next((cols_low[k] for k in AGE_KEYS if k in cols_low), None)
    proto_col = next((cols_low[k] for k in ['protocol', 'protocolo']
                      if k in cols_low), None)

    print(f"\n{'-'*90}")
    print(f"DEMOGRAFIA: sujeito='{subj_col}', sexo='{sex_col}', idade='{age_col}', "
          f"protocolo='{proto_col}'")
    print(demo.head(36).to_string())

    ren = {subj_col: 'subject', sex_col: 'sex'}
    if proto_col:
        ren[proto_col] = 'protocol_demo'
    demo = demo.rename(columns=ren)
    demo['subject'] = demo['subject'].astype(str).str.strip()
    keep = ['subject', 'sex'] + (['protocol_demo'] if proto_col else []) + \
           ([age_col] if age_col else [])
    demo = demo[keep].drop_duplicates('subject').set_index('subject')

    overlap = sorted(set(demo.index) & set(S.index))
    print(f"\nIDs que casam com as metricas: {len(overlap)} de {len(S)}")
    if len(overlap) == 0:
        print(f"  demografia: {list(demo.index[:6])}")
        print(f"  metricas:   {list(S.index[:6])}")
    else:
        M = S.join(demo, how='left')
        print("\nValores de sexo:", M['sex'].value_counts(dropna=False).to_dict())
        print(pd.crosstab(M['prefix'], M['sex']).to_string())
        if 'protocol_demo' in M.columns:
            print(pd.crosstab(M['prefix'], M['protocol_demo']).to_string())
            print(pd.crosstab(M['sex'], M['protocol_demo']).to_string())


        gs = M.groupby('sex').agg(
            n=('auc', 'size'), auc_mean=('auc', 'mean'),
            auc_median=('auc', 'median'), recall_mean=('recall', 'mean'),
            fnr_mean=('fnr', 'mean'))
        print(gs.round(3).to_string())

        sexes = [s for s in M['sex'].dropna().unique()]
        if len(sexes) == 2:
            g0 = M.loc[M['sex'] == sexes[0], 'auc']
            g1 = M.loc[M['sex'] == sexes[1], 'auc']
            u, p = mannwhitneyu(g0, g1, alternative='two-sided')
            print(f"\nMann-Whitney AUC ({sexes[0]} vs {sexes[1]}): U={u:.1f} "
                  f"p={p:.4f} -> "
                  f"{'significativo' if p < 0.05 else 'nao significativo'}")

        cc = ['auc', 'recall', 'sex', 'prefix'] + \
             (['protocol_demo'] if 'protocol_demo' in M.columns else [])
        print(M.loc[[s for s in TARGET_SUBJECTS if s in M.index], cc].to_string())
