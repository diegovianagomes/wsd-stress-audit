#%% 
#   - 4 classes montadas dos arquivos por estado; REST vem de uma fonte unica
#     (sessao de estresse), pois os REST sao distintos por sessao.
#   - StratifiedGroupKFold de 10 folds; predicoes agrupadas out-of-fold.
#   - Metricas: F1 macro e acuracia balanceada (tratam as classes igualmente
#     sob desbalanceo), com F1 por classe e a MATRIZ DE CONFUSAO agrupada como
#     peca central. A pergunta de interesse e se o estresse se confunde com os
#     estados de exercicio pela elevacao cardiaca comum.
#   - A analise por sujeito NAO se aplica ao ANAEROBIC (mediana de 3 janelas por
#     sujeito); o multiclasse e um resultado agregado.
# Comparam-se as 3 arvores com e sem balanceamento por sample_weight.


import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (f1_score, balanced_accuracy_score,
                             confusion_matrix, classification_report)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
    raise FileNotFoundError(f"{marker}")

PROJECT_ROOT = _find_project_root("src")
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PROCESSED_DIR

FIG_DIR = PROJECT_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"
GROUP = "subject_id"
COLS_TO_DROP = ['subject_id', 'window_id', 'label', 'scenario', 'protocol']
REST_SOURCE_STATE = "STRESS"
N_FOLDS = 10
SEED = 42
CLASS_ORDER = ['REST', 'STRESS', 'AEROBIC', 'ANAEROBIC']
MODELS = ['RF', 'XGB', 'LGBM']
TREATMENTS = ['none', 'balanced']

# FABRICA DE MODELOS

def make_model(name):
    if name == 'RF':
        return RandomForestClassifier(n_estimators=300, max_features=0.6,
                                      random_state=SEED, n_jobs=-1)
    if name == 'XGB':
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric='mlogloss', random_state=SEED,
                             n_jobs=-1, verbosity=0)
    if name == 'LGBM':
        return LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=SEED, n_jobs=-1, verbosity=-1)
    raise ValueError(name)


# %% MONTAGEM DO MULTICLASSE

files = sorted(Path(PROCESSED_DIR).glob("dataset_*.csv"))
scen = {}
for f in files:
    d = pd.read_csv(f)
    state = f.stem.replace('dataset_', '').upper()
    if set(d[TARGET].dropna().unique().tolist()) == {0, 1}:
        scen[state] = d

feat_sets = [set(d.columns) - set(COLS_TO_DROP) for d in scen.values()]
common_feats = sorted(set.intersection(*feat_sets))
print(f"Estados: {list(scen.keys())} | features comuns: {len(common_feats)}")

frames = []
rest = scen[REST_SOURCE_STATE]
rest = rest.loc[rest[TARGET] == 0, [GROUP] + common_feats].copy()
rest['state'] = 'REST'
frames.append(rest)
for s, d in scen.items():
    pos = d.loc[d[TARGET] == 1, [GROUP] + common_feats].copy()
    pos['state'] = s
    frames.append(pos)
multi = pd.concat(frames, ignore_index=True)

X = multi[common_feats].to_numpy()
le = LabelEncoder().fit(multi['state'])
y = le.transform(multi['state'])
groups = multi[GROUP].to_numpy()
print(f"Janelas {len(multi)} | classes {dict(multi['state'].value_counts())}")
print(f"NaN nas features: {int(np.isnan(X).sum())}")

# %% MODELO x TRATAMENTO (predicoes agrupadas out-of-fold)
sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(sgkf.split(X, y, groups))

results = []
oof_store = {}
for name in MODELS:
    for treat in TREATMENTS:
        oof = np.full(len(y), -1)
        for tr, te in folds:
            model = make_model(name)
            sw = compute_sample_weight('balanced', y[tr]) if treat == 'balanced' \
                else None
            model.fit(X[tr], y[tr], sample_weight=sw)
            oof[te] = model.predict(X[te])
        f1m = f1_score(y, oof, average='macro')
        bacc = balanced_accuracy_score(y, oof)
        oof_store[(name, treat)] = oof
        results.append({'modelo': name, 'tratamento': treat,
                        'F1_macro': f1m, 'bal_acc': bacc})
        print(f"  {name:5s} | {treat:8s} | F1_macro={f1m:.4f} | bal_acc={bacc:.4f}")

R = pd.DataFrame(results).sort_values('F1_macro', ascending=False)
print(R.round(4).to_string(index=False))

# %% MELHOR MODELO F1 POR CLASSE E MATRIZ DE CONFUSAO
best = R.iloc[0]
best_key = (best['modelo'], best['tratamento'])
oof = oof_store[best_key]
y_lbl = le.inverse_transform(y)
oof_lbl = le.inverse_transform(oof)
print(classification_report(y_lbl, oof_lbl, labels=CLASS_ORDER, digits=3,
                            zero_division=0))

cm = confusion_matrix(y_lbl, oof_lbl, labels=CLASS_ORDER)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

print(pd.DataFrame(cm, index=CLASS_ORDER, columns=CLASS_ORDER).to_string())
print(pd.DataFrame(cm_norm.round(3), index=CLASS_ORDER,
                   columns=CLASS_ORDER).to_string())

# %% FMATRIZ DE CONFUSAO NORMALIZADA
fig, ax = plt.subplots(figsize=(6.5, 5.5))
im = ax.imshow(cm_norm, cmap='Blues', vmin=0, vmax=1)
ax.set_xticks(range(len(CLASS_ORDER)))
ax.set_yticks(range(len(CLASS_ORDER)))
ax.set_xticklabels(CLASS_ORDER, rotation=30, ha='right')
ax.set_yticklabels(CLASS_ORDER)
ax.set_xlabel("Predito")
ax.set_ylabel("Verdadeiro")
ax.set_title(f"Matriz de confusao normalizada ({best['modelo']}, {best['tratamento']})")
for i in range(len(CLASS_ORDER)):
    for j in range(len(CLASS_ORDER)):
        ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha='center', va='center',
                color='white' if cm_norm[i, j] > 0.5 else '#222', fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
plt.tight_layout()
fig_path = FIG_DIR / "multiclasse_matriz_confusao.png"
plt.savefig(fig_path, dpi=150)
print(f"\n{fig_path}")
