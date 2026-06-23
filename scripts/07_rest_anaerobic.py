#%% 

#   - StratifiedGroupKFold de 10 folds (respeita sujeitos e equilibra a classe
#     rara entre folds; nenhum fold fica sem positivo).
#   - Metrica principal: AUPRC AGRUPADA out-of-fold. As predicoes de teste de
#     todos os folds sao concatenadas e a AUPRC e calculada uma unica vez sobre
#     os 89 positivos. E a estimativa mais estavel sob escassez. AUPRC por fold
#     entra so como dispersao, e a AUC fica como secundaria.
#   - Comparacao de quatro tratamentos de desbalanceamento (none, class_weight,
#     RandomUnder, SMOTE) nas tres arvores (RF, XGB, LGBM), com o ganho julgado
#     pela AUPRC agrupada e nao por deslocamento de limiar.
#

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
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
    raise FileNotFoundError(f"Rnada foi encontrado '{marker}').")

PROJECT_ROOT = _find_project_root("src")
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PROCESSED_DIR

DATA_PATH = PROCESSED_DIR / "dataset_anaerobic.csv"
FIG_DIR = PROJECT_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"
GROUP = "subject_id"
COLS_TO_DROP = ['subject_id', 'window_id', 'label', 'scenario', 'protocol']
N_FOLDS = 10
SEED = 42
TREATMENTS = ['none', 'class_weight', 'RandomUnder', 'SMOTE']
MODELS = ['RF', 'XGB', 'LGBM']



def make_estimator(model_name, treatment, scale_pos_weight):
    """Monta o estimador para um modelo e um tratamento de desbalanceamento.
    Resampling so atua no treino (Pipeline do imbalanced-learn). class_weight e
    parametro do modelo (scale_pos_weight no XGB)."""
    cw = (treatment == 'class_weight')

    if model_name == 'RF':
        clf = RandomForestClassifier(
            n_estimators=300, max_features=0.6, random_state=SEED, n_jobs=-1,
            class_weight=('balanced' if cw else None))
    elif model_name == 'XGB':
        clf = XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, eval_metric='logloss',
            random_state=SEED, n_jobs=-1, verbosity=0,
            scale_pos_weight=(scale_pos_weight if cw else 1))
    elif model_name == 'LGBM':
        clf = LGBMClassifier(
            n_estimators=300, num_leaves=31, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=SEED, n_jobs=-1,
            verbosity=-1, class_weight=('balanced' if cw else None))
    else:
        raise ValueError(model_name)

    if treatment == 'RandomUnder':
        return ImbPipeline([('res', RandomUnderSampler(random_state=SEED)),
                            ('clf', clf)])
    if treatment == 'SMOTE':
        return ImbPipeline([('res', SMOTE(random_state=SEED, k_neighbors=5)),
                            ('clf', clf)])
    return clf


# %% CARGA E PARTICIONAMENTO

data = pd.read_csv(DATA_PATH)
feature_cols = [c for c in data.columns if c not in COLS_TO_DROP]
X = data[feature_cols].to_numpy()
y = data[TARGET].to_numpy().astype(int)
groups = data[GROUP].to_numpy()

n_pos, n_neg = int(y.sum()), int((1 - y).sum())
prevalencia = n_pos / len(y)
spw = n_neg / n_pos
print(f"Janelas {len(y)} | positivos {n_pos} | negativos {n_neg} | "
      f"prevalencia {prevalencia:.4f}")
print(f"Piso de AUPRC (no-skill) = {prevalencia:.4f} | scale_pos_weight = {spw:.1f}")

sgkf = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
folds = list(sgkf.split(X, y, groups))

# %%  MODELO x TRATAMENTO (predicoes agrupadas out-of-fold)
results = []
oof_store = {}

for model_name in MODELS:
    for treat in TREATMENTS:
        oof = np.full(len(y), np.nan)
        fold_ap = []
        try:
            for tr_idx, te_idx in folds:
                est = make_estimator(model_name, treat, spw)
                est.fit(X[tr_idx], y[tr_idx])
                p = est.predict_proba(X[te_idx])[:, 1]
                oof[te_idx] = p
                fold_ap.append(average_precision_score(y[te_idx], p))
            ap_pooled = average_precision_score(y, oof)
            auc_pooled = roc_auc_score(y, oof)
            oof_store[(model_name, treat)] = oof
            results.append({'modelo': model_name, 'tratamento': treat,
                            'AUPRC_pool': ap_pooled, 'AUC_pool': auc_pooled,
                            'AUPRC_fold_mean': np.mean(fold_ap),
                            'AUPRC_fold_std': np.std(fold_ap)})
            print(f"  {model_name:5s} | {treat:12s} | AUPRC_pool={ap_pooled:.4f} "
                  f"| AUC_pool={auc_pooled:.4f} | AUPRC_fold={np.mean(fold_ap):.4f}"
                  f"+/-{np.std(fold_ap):.4f}")
        except Exception as e:
            print(f"  {model_name:5s} | {treat:12s} | FALHOU: {e}")

R = pd.DataFrame(results)

# %% RESULTADOS ORDENADOS POR AUPRC AGRUPADA

R_sorted = R.sort_values('AUPRC_pool', ascending=False)
print(R_sorted.round(4).to_string(index=False))
print(f"\nPiso de no-skill = {prevalencia:.4f}.")

# GANHO DE AUPRC AGRUPADA SOBRE 'none', POR MODELO
for model_name in MODELS:
    sub = R[R['modelo'] == model_name].set_index('tratamento')['AUPRC_pool']
    if 'none' not in sub.index:
        continue
    base = sub['none']
    deltas = (sub - base).drop('none')
    txt = " | ".join(f"{t}={d:+.4f}" for t, d in deltas.items())
    print(f"  {model_name:5s} | none={base:.4f} | {txt}")

# %% CURVA PRECISION-RECALL DO MELHOR MODELO
best_model = R.loc[R['AUPRC_pool'].idxmax()]
print(f"\n{best_model}")

plt.figure(figsize=(7.5, 6))
for treat in TREATMENTS:
    key = (best_model, treat)
    if key not in oof_store:
        continue
    prec, rec, _ = precision_recall_curve(y, oof_store[key])
    ap = R[(R.modelo == best_model) & (R.tratamento == treat)]['AUPRC_pool'].iloc[0]
    plt.plot(rec, prec, lw=1.8, label=f"{treat} (AUPRC={ap:.3f})")
plt.axhline(prevalencia, color='#b00', ls='--', lw=1,
            label=f"no-skill ({prevalencia:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"REST vs ANAEROBIC  curvas PR agrupadas ({best_model})")
plt.legend(loc='upper right', fontsize=9)
plt.grid(alpha=.25)
plt.tight_layout()
fig_path = FIG_DIR / "anaerobic_pr_curves.png"
plt.savefig(fig_path, dpi=150)
print(f"Figura salva em: {fig_path}")

