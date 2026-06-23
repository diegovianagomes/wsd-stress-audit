#%% 
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from lightgbm import LGBMClassifier
import shap
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
    raise FileNotFoundError(f"Raiz do projeto nao encontrada (marcador '{marker}').")

PROJECT_ROOT = _find_project_root("src")
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PROCESSED_DIR

DATA_PATH = PROCESSED_DIR / "dataset_stress.csv"
FIG_DIR = PROJECT_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"
GROUP = "subject_id"
COLS_TO_DROP = ['subject_id', 'window_id', 'label', 'scenario', 'protocol']
N_FOLDS = 10
SEED = 42

DISSOCIADOS = ['f02', 'f03', 'f04', 'f18']
FACEIS = ['f01', 'S08']

LGBM_PARAMS = dict(n_estimators=300, num_leaves=31, learning_rate=0.05,
                   subsample=0.8, colsample_bytree=0.8, random_state=SEED,
                   n_jobs=-1, verbosity=-1)

EDA_TOKENS = ['eda', 'phasic', 'tonic', 'peaks', 'scl', 'scr']
CARD_TOKENS = ['hr_', 'rmssd', 'sdnn', 'pnn', 'hf', 'lf', 'bvp', 'ibi']

# %% CARGA E SHAP OUT-OF-FOLD


data = pd.read_csv(DATA_PATH)
feature_cols = [c for c in data.columns if c not in COLS_TO_DROP]
X = data[feature_cols].to_numpy()
y = data[TARGET].to_numpy().astype(int)
groups = data[GROUP].to_numpy()
n, p = X.shape
print(f"Janelas {n} | features {p} | sujeitos {data[GROUP].nunique()}")

shap_full = np.full((n, p), np.nan)
proba_full = np.full(n, np.nan)

gkf = GroupKFold(n_splits=N_FOLDS)
for fold, (tr, te) in enumerate(gkf.split(X, y, groups), start=1):
    model = LGBMClassifier(**LGBM_PARAMS).fit(X[tr], y[tr])
    proba_full[te] = model.predict_proba(X[te])[:, 1]
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X[te], check_additivity=False)
    if isinstance(sv, list):
        sv = sv[1]
    shap_full[te] = sv
    print(f"  fold {fold:2d} ok")

# %% IMPORTANCIA GLOBAL 
imp = pd.Series(np.nanmean(np.abs(shap_full), axis=0),
                index=feature_cols).sort_values(ascending=False)
print(imp.head(15).round(4).to_string())

# %% AGRUPAMENTO POR CANAL
def match(tokens):
    return [i for i, f in enumerate(feature_cols)
            if any(t in f.lower() for t in tokens)]

eda_idx = match(EDA_TOKENS)
card_idx = match(CARD_TOKENS)
print(f"\nFeatures EDA ({len(eda_idx)}): {[feature_cols[i] for i in eda_idx]}")
print(f"Features cardiacas ({len(card_idx)}): {[feature_cols[i] for i in card_idx]}")

shap_eda = shap_full[:, eda_idx].sum(axis=1)
shap_card = shap_full[:, card_idx].sum(axis=1)
df = pd.DataFrame({'subject': groups, 'label': y, 'proba': proba_full,
                   'shap_eda': shap_eda, 'shap_card': shap_card})

# %% DIRECAO DAS ATRIBUICOES POR SUJEITO (janelas de estresse)

rows = []
for s in DISSOCIADOS + FACEIS:
    sub = df[df['subject'] == s]
    st = sub[sub['label'] == 1]
    rest = sub[sub['label'] == 0]
    eda_st, eda_re = st['shap_eda'].mean(), rest['shap_eda'].mean()
    card_st, card_re = st['shap_card'].mean(), rest['shap_card'].mean()
    rows.append({
        'subject': s,
        'grupo': 'dissociado' if s in DISSOCIADOS else 'facil',
        'n_stress': len(st),
        'EDA_sep': eda_st - eda_re,
        'CARD_sep': card_st - card_re,
        'EDA_stress': eda_st, 'EDA_rest': eda_re,
        'proba_stress': st['proba'].mean()})
T = pd.DataFrame(rows)
print(T.round(3).to_string(index=False))

# %% SHAP de EDA nas janelas de estresse, por sujeito
order = T.sort_values('EDA_sep')
colors = ['#b00' if g == 'dissociado' else '#3b6ea5' for g in order['grupo']]
plt.figure(figsize=(8, 5))
plt.bar(order['subject'], order['EDA_sep'], color=colors)
plt.axhline(0, color='#333', lw=1)
plt.ylabel("Separacao SHAP da EDA (estresse - repouso)")
plt.title("Quanto a EDA diferencia estresse de repouso por sujeito "
          "(vermelho = dissociado)")
plt.grid(axis='y', alpha=.25)
plt.tight_layout()
fig_path = FIG_DIR / "shap_separacao_eda_por_sujeito.png"
plt.savefig(fig_path, dpi=150)
print(f"\nFigura salva em: {fig_path}")

# %% BEESWARM GLOBAL
try:
    plt.figure()
    shap.summary_plot(shap_full, data[feature_cols], show=False, max_display=15)
    plt.tight_layout()
    bee_path = FIG_DIR / "shap_beeswarm_global.png"
    plt.savefig(bee_path, dpi=150, bbox_inches='tight')
    print(f"Beeswarm global salvo em: {bee_path}")
except Exception as e:
    print(f"Beeswarm pulado: {e}")
