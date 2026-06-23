#%% 
# PARTE 1 - varredura de CALIB_FRAC:
#   O salto do f18 (AUC de ~0.49 para ~0.90 com delta calibrada) foi medido com
#   metade do REST como enrollment. Aqui varia-se a fracao de calibracao para
#   ver se o ganho e estavel ou artefato do split. Se o AUC do f18 se mantem
#   alto em 0.2, 0.3, 0.5 e 0.7, e mascaramento por offset de baseline de fato.
#
# PARTE 2 - Assinatura autonomica :
#   Para cada sujeito de interesse, mede-se duas coisas por feature autonomica.
#   (a) Separacao INTRA-sujeito REST vs STRESS, via d de Cohen. Diz se o sinal
#       de estresse existe dentro do proprio sujeito.
#   (b) OFFSET de baseline: distancia do REST medio do sujeito ao REST medio
#       global, em desvios-padrao. Diz se o nivel absoluto dele e atipico.



import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
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
N_FOLDS = 10
TARGET = "label"
COLS_TO_DROP = ['subject_id', 'window_id', 'label', 'scenario', 'protocol']
TARGET_SUBJECTS = ['f02', 'f03', 'f04', 'f18']
EASY_SUBJECTS = ['f01', 'S08']
RF_PARAMS = dict(n_estimators=250, max_features=0.6, random_state=42, n_jobs=-1)
SEED = 42
KEY_FEATURES = ['hr_mean', 'rmssd', 'pnn50', 'mean_raw_eda', 'mean_tonic_eda',
                'bvp_std', 'sdnn', 'HF_n', 'ratio', 'acc_std', 'peaks_density']
CALIB_FRACS = [0.2, 0.3, 0.5, 0.7]
EDA_FEATURES = ['mean_tonic_eda', 'mean_raw_eda', 'std_phasic_eda',
                'std_tonic_eda', 'peaks_density']
CARDIAC_FEATURES = ['hr_mean', 'rmssd', 'sdnn', 'pnn50', 'bvp_std', 'HF_n']


# FUNCOES



def build_train_delta(train, key_features):
    """Delta no treino: baseline = REST do proprio sujeito de treino.
    Nao depende da fracao de calibracao."""
    tr = train.copy()
    for subj in tr['subject_id'].unique():
        rest = tr[(tr['subject_id'] == subj) & (tr[TARGET] == 0)][key_features]
        base = rest.mean() if len(rest) else \
            tr[tr['subject_id'] == subj][key_features].mean()
        m = tr['subject_id'] == subj
        for f in key_features:
            denom = abs(base[f]) if base[f] != 0 else 1.0
            tr.loc[m, f'delta_{f}'] = (tr.loc[m, f] - base[f]) / (denom + 1e-12)
    return tr


def build_test_delta(test, key_features, calib_frac, seed=SEED):
    """Delta no teste: baseline = parte do REST do proprio sujeito (enrollment).
    As janelas de calibracao saem da avaliacao. Depende da fracao."""
    rng = np.random.RandomState(seed)
    te = test.copy()
    eval_idx = []
    for subj in te['subject_id'].unique():
        sub = te[te['subject_id'] == subj]
        rest_idx = sub[sub[TARGET] == 0].index.tolist()
        rng.shuffle(rest_idx)
        n_calib = max(1, int(len(rest_idx) * calib_frac))
        calib = set(rest_idx[:n_calib])
        base = te.loc[list(calib), key_features].mean()
        m = te['subject_id'] == subj
        for f in key_features:
            denom = abs(base[f]) if base[f] != 0 else 1.0
            te.loc[m, f'delta_{f}'] = (te.loc[m, f] - base[f]) / (denom + 1e-12)
        eval_idx.extend([i for i in sub.index if i not in calib])
    return te, sorted(eval_idx)


def cohens_d(x, y):
    """d de Cohen entre x (stress) e y (rest). Positivo = stress maior."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return np.nan
    vx, vy = x.var(ddof=1), y.var(ddof=1)
    sp = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    return (x.mean() - y.mean()) / sp if sp > 0 else np.nan



# %% VARREDURA DE CALIB_FRAC f18
data = pd.read_csv(DATA_PATH)
gkf = GroupKFold(n_splits=N_FOLDS)
fold_splits = list(gkf.split(data, data[TARGET], data['subject_id']))
feature_cols = [c for c in data.columns if c not in COLS_TO_DROP]
key_feats = [f for f in KEY_FEATURES if f in feature_cols]
delta_cols = [f'delta_{f}' for f in key_feats]

sweep = []
for fold, (tr_idx, te_idx) in enumerate(fold_splits, start=1):
    train = data.iloc[tr_idx].reset_index(drop=True)
    test = data.iloc[te_idx].reset_index(drop=True)
    here = [s for s in TARGET_SUBJECTS if s in set(test['subject_id'])]
    if not here:
        continue

    tr_d = build_train_delta(train, key_feats)
    feat_d = feature_cols + delta_cols
    m_delta = RandomForestClassifier(**RF_PARAMS).fit(
        tr_d[feat_d].fillna(0), train[TARGET])
    m_base = RandomForestClassifier(**RF_PARAMS).fit(
        train[feature_cols], train[TARGET])

    # AUC baseline (absoluto) por sujeito, no teste completo, como referencia
    base_auc = {}
    for s in here:
        msk = (test['subject_id'] == s).values
        yt = test.loc[msk, TARGET].values
        if len(np.unique(yt)) > 1:
            pr = m_base.predict_proba(test.loc[msk, feature_cols])[:, 1]
            base_auc[s] = roc_auc_score(yt, pr)
        else:
            base_auc[s] = np.nan

    for frac in CALIB_FRACS:
        te_d, eval_idx = build_test_delta(test, key_feats, frac)
        esubj = te_d.loc[eval_idx, 'subject_id'].values
        yev = te_d.loc[eval_idx, TARGET].values
        prd = m_delta.predict_proba(te_d.loc[eval_idx, feat_d].fillna(0))[:, 1]
        for s in here:
            m = (esubj == s)
            yt = yev[m]
            auc = roc_auc_score(yt, prd[m]) if len(np.unique(yt)) > 1 else np.nan
            sweep.append({'subject': s, 'calib_frac': frac,
                          'auc_base': base_auc[s], 'auc_delta': auc,
                          'n_eval': int(m.sum())})

SW = pd.DataFrame(sweep)
piv = SW.pivot_table(index='subject', columns='calib_frac', values='auc_delta')
ref = SW.groupby('subject')['auc_base'].first().rename('auc_base')
out1 = ref.to_frame().join(piv)
print(out1.round(3).to_string())



# %% ASSINATURA AUTONOMICA (Cohen d intra-sujeito e offset)


auto_feats = [f for f in (EDA_FEATURES + CARDIAC_FEATURES) if f in data.columns]
print(f"Features autonomicas utilizadas({len(auto_feats)}): {auto_feats}")

subjects = TARGET_SUBJECTS + EASY_SUBJECTS
rest_all = data[data[TARGET] == 0]
glob_rest_mean = rest_all[auto_feats].mean()
glob_rest_std = rest_all[auto_feats].std().replace(0, 1)

d_rows, off_rows = [], []
for s in subjects:
    sd = data[data['subject_id'] == s]
    rest = sd[sd[TARGET] == 0]
    stress = sd[sd[TARGET] == 1]
    drow = {'subject': s}
    orow = {'subject': s}
    for f in auto_feats:
        drow[f] = cohens_d(stress[f], rest[f])               # sinal intra-sujeito
        orow[f] = (rest[f].mean() - glob_rest_mean[f]) / glob_rest_std[f]  # offset
    d_rows.append(drow)
    off_rows.append(orow)

D = pd.DataFrame(d_rows).set_index('subject')
OFF = pd.DataFrame(off_rows).set_index('subject')

print(D.round(2).to_string())
print(OFF.round(2).to_string())
