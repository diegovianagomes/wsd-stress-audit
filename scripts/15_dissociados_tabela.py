#%% 
# Cruzar, por sujeito, os valores
# SHAP com o autorrelato e as caracteristicas autonomicas. Cada coluna e
# recomputada no MESMO pipeline subject-independent (GroupKFold, LGBM, out-of-
# fold).
#
# Sujeitos: f02, f03, f04, f18 (dissociados corrigidos) e f01 como referencia
# de sujeito bem classificado, para contraste.
#
# Colunas:
#   mecanismo         : leitura corrigida da Etapa 17
#   autorrelato_rest  : estresse subjetivo medio em repouso (0-10)
#   z_eda_ton_rest    : EDA tonica em repouso, z relativo a coorte
#   z_scr_react       : delta SCR (stress-rest), z relativo a coorte
#   z_rmssd_react     : delta RMSSD, z relativo a coorte
#   z_pnn50_react     : delta pNN50, z relativo a coorte
#   d_eda_ton         : Cohen d intra-sujeito, EDA tonica REST vs STRESS
#   d_pnn50           : Cohen d intra-sujeito, pNN50 REST vs STRESS
#   shap_sep_eda      : separacao SHAP do canal EDA (stress - rest), out-of-fold
#   shap_sep_card     : separacao SHAP do canal cardiaco (stress - rest)
#   auc_RF/XGB/LGBM   : AUC por sujeito com 49 features, um por modelo
#   auc_LGBM_semACC   : AUC por sujeito (LGBM) sem as features de ACC
#   recall_05_LGBM    : recall de STRESS no limiar 0.5 (LGBM, full)================

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr  # noqa: F401 (mantido p/ extensoes)
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False

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

OUT_DIR = PROJECT_ROOT / "experiments" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"
GROUP = "subject_id"
COLS_DROP = ['subject_id', 'window_id', 'label', 'scenario', 'protocol']
TARGETS = ['f02', 'f03', 'f04', 'f18']
REFERENCE = ['f01']
ROWS = TARGETS + REFERENCE
N_FOLDS = 10
SEED = 42

REPORT_REST_COLS = ['Baseline', 'First Rest', 'Second Rest']
EDA_FEATS = ['mean_raw_eda', 'std_raw_eda', 'mean_tonic_eda', 'std_tonic_eda',
             'tonic_ratio_up', 'tonic_ratio_down', 'mean_phasic_eda',
             'std_phasic_eda', 'peaks_density', 'scr_mean_amp', 'scr_mean_height',
             'scr_mean_risetime', 'scr_mean_recoverytime']
CARD_FEATS = ['hr_mean', 'hr_std', 'rmssd', 'sdnn', 'pnn20', 'pnn50', 'mean_ibi',
              'max_ibi', 'min_ibi', 'bvp_std', 'HF_peak', 'HF_n', 'LF_n',
              'VLF_power', 'LF_power', 'HF_power', 'ratio', 'total_power',
              'VHF_peak']
MECANISMO = {
    'f02': "conflito de canais (EDA sobe, HRV inverte)",
    'f03': "invertido/refratario (EDA cai no estresse)",
    'f04': "limiar (assinatura limpa, mal limiarizada)",
    'f18': "conflito brando (EDA forte, FC cai); baseline nao elevado",
    'f01': "referencia: bem classificado (separa pela EDA)",
}


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


# %% DADOS

data = pd.read_csv(Path(PROCESSED_DIR) / "dataset_stress.csv").reset_index(drop=True)
feat_cols = [c for c in data.columns if c not in COLS_DROP]
acc_cols = [c for c in feat_cols if 'acc' in c.lower()]
noacc_cols = [c for c in feat_cols if c not in acc_cols]
EDA_FEATS = [f for f in EDA_FEATS if f in feat_cols]
CARD_FEATS = [f for f in CARD_FEATS if f in feat_cols]

y = data[TARGET].to_numpy()
groups = data[GROUP].to_numpy()
gkf = GroupKFold(n_splits=N_FOLDS)
folds = list(gkf.split(data, y, groups))
print(f"Janelas {len(data)} | sujeitos {data[GROUP].nunique()} | "
      f"features {len(feat_cols)} (ACC={len(acc_cols)}) | SHAP={'sim' if SHAP_OK else 'nao'}")

# %% OOF: PROBA (full e sem ACC) e SHAP (full)

def oof_proba(cols, name):
    X = data[cols].to_numpy()
    p = np.full(len(y), np.nan)
    for tr, te in folds:
        m = make_model(name)
        m.fit(X[tr], y[tr])
        p[te] = m.predict_proba(X[te])[:, 1]
    return p


p_full = {name: oof_proba(feat_cols, name) for name in ['RF', 'XGB', 'LGBM']}
p_noacc = oof_proba(noacc_cols, 'LGBM')

shap_oof = None
if SHAP_OK:
    Xf = data[feat_cols].to_numpy()
    shap_oof = np.full((len(y), len(feat_cols)), np.nan)
    for tr, te in folds:
        m = make_model('LGBM')
        m.fit(Xf[tr], y[tr])
        expl = shap.TreeExplainer(m)
        sv = expl.shap_values(Xf[te])
        if isinstance(sv, list):
            sv = sv[1]
        shap_oof[te] = sv
    shap_df = pd.DataFrame(shap_oof, columns=feat_cols)


rest_mask = data[TARGET] == 0
stress_mask = data[TARGET] == 1


def per_subject_auc(p, subj):
    m = groups == subj
    if len(np.unique(y[m])) < 2:
        return np.nan
    return roc_auc_score(y[m], p[m])


def recall_at(p, subj, thr=0.5):
    m = (groups == subj) & (y == 1)
    if m.sum() == 0:
        return np.nan
    return float((p[m] >= thr).mean())


def z_rest_eda_tonic():
    s = data[rest_mask].groupby(GROUP)['mean_tonic_eda'].mean()
    oth = s[~s.index.isin(TARGETS)]
    return (s - oth.mean()) / oth.std()


def z_react(feat):
    d = (data[stress_mask].groupby(GROUP)[feat].mean()
         - data[rest_mask].groupby(GROUP)[feat].mean())
    oth = d[~d.index.isin(TARGETS)]
    return (d - oth.mean()) / oth.std()


def cohen_d_within(subj, feat):
    r = data[(groups == subj) & (y == 0)][feat].dropna()
    s = data[(groups == subj) & (y == 1)][feat].dropna()
    if len(r) < 2 or len(s) < 2:
        return np.nan
    pooled = np.sqrt((r.std() ** 2 + s.std() ** 2) / 2)
    return (s.mean() - r.mean()) / pooled if pooled > 0 else np.nan


def shap_sep(subj, feats):
    if shap_oof is None:
        return np.nan
    mr = (groups == subj) & (y == 0)
    ms = (groups == subj) & (y == 1)
    if mr.sum() == 0 or ms.sum() == 0:
        return np.nan
    return float(shap_df.loc[ms, feats].mean().sum()
                 - shap_df.loc[mr, feats].mean().sum())


# self-report
sr = {}
for f in PROJECT_ROOT.rglob("Stress_Level_v*.csv"):
    try:
        df = pd.read_csv(f, index_col=0)
    except Exception:
        continue
    cols = [c for c in REPORT_REST_COLS if c in df.columns]
    if cols:
        for s in df.index:
            sr[s] = float(df.loc[s, cols].mean())

zeda = z_rest_eda_tonic()
zscr = z_react('scr_mean_amp')
zrmssd = z_react('rmssd')
zpnn = z_react('pnn50')

# %% MONTAGEM DA TABELA
rows = []
for s in ROWS:
    rows.append({
        'sujeito': s,
        'mecanismo': MECANISMO.get(s, ''),
        'n_rest': int(((groups == s) & (y == 0)).sum()),
        'n_stress': int(((groups == s) & (y == 1)).sum()),
        'autorrelato_rest': round(sr.get(s, np.nan), 2),
        'z_eda_ton_rest': round(float(zeda.get(s, np.nan)), 2),
        'z_scr_react': round(float(zscr.get(s, np.nan)), 2),
        'z_rmssd_react': round(float(zrmssd.get(s, np.nan)), 2),
        'z_pnn50_react': round(float(zpnn.get(s, np.nan)), 2),
        'd_eda_ton': round(cohen_d_within(s, 'mean_tonic_eda'), 2),
        'd_pnn50': round(cohen_d_within(s, 'pnn50'), 2),
        'shap_sep_eda': round(shap_sep(s, EDA_FEATS), 3),
        'shap_sep_card': round(shap_sep(s, CARD_FEATS), 3),
        'auc_RF': round(per_subject_auc(p_full['RF'], s), 3),
        'auc_XGB': round(per_subject_auc(p_full['XGB'], s), 3),
        'auc_LGBM': round(per_subject_auc(p_full['LGBM'], s), 3),
        'auc_LGBM_semACC': round(per_subject_auc(p_noacc, s), 3),
        'recall_05_LGBM': round(recall_at(p_full['LGBM'], s), 3),
    })

tab = pd.DataFrame(rows).set_index('sujeito')

print(tab.T.to_string())

csv_path = OUT_DIR / "tabela_consolidada_dissociados.csv"
tab.to_csv(csv_path)
print(f"\nTabela salva em: {csv_path}")

aucs = tab[['auc_RF', 'auc_XGB', 'auc_LGBM']]
spread = aucs.max(axis=1) - aucs.min(axis=1)
for s in ROWS:
    print(f"  {s}: min={aucs.loc[s].min():.3f} max={aucs.loc[s].max():.3f} "
          f"amplitude={spread[s]:.3f}")
