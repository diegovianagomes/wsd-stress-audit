#%% 
# O multiclasse anterior mostrou que o ACC separa perfeitamente
# sentado de exercicio, confirmando que ele e um canal de ATIVIDADE. Como REST
# e STRESS sao ambos sentados, a pergunta de sensibilidade e: a deteccao de
# estresse se apoia no ACC, ou no sinal autonomico (EDA e cardiaco)?
#   1. AUC agregado (predicoes agrupadas out-of-fold) nas 3 arvores.
#   2. AUC por sujeito (LGBM), para verificar se remover o ACC muda QUEM e
#      dissociado. Se a dissociacao de f02/f03/f04/f18 persiste sem o ACC, ela
#      e genuinamente autonomica, e nao efeito do canal de movimento.



import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
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
    raise FileNotFoundError(f"Nope '{marker}').")

PROJECT_ROOT = _find_project_root("src")
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PROCESSED_DIR

FIG_DIR = PROJECT_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"
GROUP = "subject_id"
COLS_TO_DROP = ['subject_id', 'window_id', 'label', 'scenario', 'protocol']
N_FOLDS = 10
SEED = 42
MODELS = ['RF', 'XGB', 'LGBM']
PRIMARY = 'LGBM'
DISSOCIADOS = ['f02', 'f03', 'f04', 'f18']
CHANCE = 0.5


def make_model(name):
    if name == 'RF':
        return RandomForestClassifier(n_estimators=300, max_features=0.6,
                                      random_state=SEED, n_jobs=-1)
    if name == 'XGB':
        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                             subsample=0.8, colsample_bytree=0.8,
                             eval_metric='logloss', random_state=SEED,
                             n_jobs=-1, verbosity=0)
    if name == 'LGBM':
        return LGBMClassifier(n_estimators=300, num_leaves=31, learning_rate=0.05,
                              subsample=0.8, colsample_bytree=0.8,
                              random_state=SEED, n_jobs=-1, verbosity=-1)
    raise ValueError(name)


# %% DADOS E SELECAO DE FEATURES

data = pd.read_csv(Path(PROCESSED_DIR) / "dataset_stress.csv")
feature_cols = [c for c in data.columns if c not in COLS_TO_DROP]
acc_cols = [c for c in feature_cols if 'acc' in c.lower()]
noacc_cols = [c for c in feature_cols if c not in acc_cols]

print(f"\nJanelas {len(data)} | sujeitos {data[GROUP].nunique()} | "
      f"label {data[TARGET].value_counts().to_dict()}")
print(f"Features totais: {len(feature_cols)}")
print(f"Features de ACC removidas ({len(acc_cols)}): {acc_cols}")
print(f"Features restantes sem ACC: {len(noacc_cols)}")

y = data[TARGET].to_numpy()
groups = data[GROUP].to_numpy()
gkf = GroupKFold(n_splits=N_FOLDS)
folds = list(gkf.split(data, y, groups))

FEATURE_SETS = {'full': feature_cols, 'sem_ACC': noacc_cols}


def oof_proba(cols, model_name):
    """Probabilidade da classe positiva agrupada out-of-fold para um conjunto
    de colunas e um modelo. Cada janela e predita uma unica vez, pelo fold em
    que ficou no teste."""
    X = data[cols].to_numpy()
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        m = make_model(model_name)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


# %% AGREGADO: AUC COM E SEM ACC, NAS 3 ARVORES

proba_cache = {}
rows = []
for name in MODELS:
    aucs = {}
    for fs_name, cols in FEATURE_SETS.items():
        p = oof_proba(cols, name)
        proba_cache[(name, fs_name)] = p
        aucs[fs_name] = roc_auc_score(y, p)
    rows.append({'modelo': name, 'AUC_full': aucs['full'],
                 'AUC_sem_ACC': aucs['sem_ACC'],
                 'delta': aucs['sem_ACC'] - aucs['full']})
agg = pd.DataFrame(rows)
print(agg.round(4).to_string(index=False))


# %% POR SUJEITO (LGBM) A DISSOCIACAO PERSISTE SEM ACC?

p_full = proba_cache[(PRIMARY, 'full')]
p_noacc = proba_cache[(PRIMARY, 'sem_ACC')]


def per_subject_auc(p):
    out = {}
    for s in pd.unique(groups):
        m = groups == s
        ys = y[m]
        if len(np.unique(ys)) < 2:
            continue
        out[s] = roc_auc_score(ys, p[m])
    return out


auc_full_s = per_subject_auc(p_full)
auc_noacc_s = per_subject_auc(p_noacc)
common = [s for s in auc_full_s if s in auc_noacc_s]

tab = pd.DataFrame({
    'sujeito': common,
    'AUC_full': [auc_full_s[s] for s in common],
    'AUC_sem_ACC': [auc_noacc_s[s] for s in common],
})
tab['delta'] = tab['AUC_sem_ACC'] - tab['AUC_full']
tab = tab.sort_values('AUC_full').reset_index(drop=True)

print(tab[tab['sujeito'].isin(DISSOCIADOS)].round(3).to_string(index=False))


n_below_full = int((tab['AUC_full'] < CHANCE).sum())
n_below_noacc = int((tab['AUC_sem_ACC'] < CHANCE).sum())
rho, pval = spearmanr(tab['AUC_full'], tab['AUC_sem_ACC'])
print(f"  Sujeitos abaixo do acaso (AUC<0.5): full={n_below_full} | "
      f"sem_ACC={n_below_noacc} (de {len(tab)})")
print(f"  Spearman entre AUC_full e AUC_sem_ACC: rho={rho:.3f} (p={pval:.1e})")
print(f"  |delta| mediano por sujeito: {tab['delta'].abs().median():.3f} | "
      f"maximo: {tab['delta'].abs().max():.3f}")


# %% AUC POR SUJEITO, FULL vs SEM ACC
fig, ax = plt.subplots(figsize=(6.2, 6))
is_dis = tab['sujeito'].isin(DISSOCIADOS).to_numpy()
ax.scatter(tab.loc[~is_dis, 'AUC_full'], tab.loc[~is_dis, 'AUC_sem_ACC'],
           c='#6699cc', s=45, label='demais sujeitos', zorder=3)
ax.scatter(tab.loc[is_dis, 'AUC_full'], tab.loc[is_dis, 'AUC_sem_ACC'],
           c='#cc3333', s=70, label='dissociados', zorder=4)
for _, r in tab[is_dis].iterrows():
    ax.annotate(r['sujeito'], (r['AUC_full'], r['AUC_sem_ACC']),
                textcoords='offset points', xytext=(6, 4), fontsize=9,
                color='#cc3333')
lim = [min(tab['AUC_full'].min(), tab['AUC_sem_ACC'].min()) - 0.03,
       max(tab['AUC_full'].max(), tab['AUC_sem_ACC'].max()) + 0.03]
ax.plot(lim, lim, '--', color='#888', lw=1, label='identidade', zorder=2)
ax.axhline(CHANCE, color='#bbb', lw=0.8, zorder=1)
ax.axvline(CHANCE, color='#bbb', lw=0.8, zorder=1)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel(f"AUC por sujeito ({PRIMARY}, todas as features)")
ax.set_ylabel(f"AUC por sujeito ({PRIMARY}, sem ACC)")
ax.set_title("Ablacao do ACC AUC por sujeito permanece sobre a identidade")
ax.legend(loc='lower right', fontsize=9)
plt.tight_layout()
fig_path = FIG_DIR / "ablacao_acc_auc_por_sujeito.png"
plt.savefig(fig_path, dpi=150)
print(f"\nFigura salva em: {fig_path}")

