#%% 
# Objetivo: testar features que ajudem nos sujeitos cujo sinal autonomico
# periferico nao separa estresse de repouso (dissociacao subjetivo-fisiologica),
# identificados na analise por sujeito: f02, f03, f04, f18.
#
#   PARADIGMA A (subject-independent, honesto):
#       Features que não precisam de baseline por sujeito. Razoes e coeficientes
#       de variacao, calculados dentro de cada janela. Validos para sujeito
#       nunca visto. Avaliados no teste completo dos folds disjuntos.
#
#   PARADIGMA B (subject-adaptive, com calibracao):
#       Delta em relacao ao baseline REST do PROPRIO sujeito de teste. Exige
#       enrollment: parte das janelas REST do sujeito de teste e separada para
#       estimar o baseline e NAO entra na avaliacao. Quantifica o teto que a
#       personalizacao alcancaria se fosse possivel coletar baseline.


#%%
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

#%%
def _find_project_root(marker="src"):
    try:
        start = Path(__file__).resolve()
    except NameError:
        start = Path.cwd().resolve()
    for parent in [start, *start.parents]:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(f"Não encontrad'{marker}').")

PROJECT_ROOT = _find_project_root("src")
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PROCESSED_DIR

DATA_PATH = PROCESSED_DIR / "dataset_stress.csv"
N_FOLDS = 10
TARGET = "label"
COLS_TO_DROP = ['subject_id', 'window_id', 'label', 'scenario', 'protocol']

# Sujeitos dissociados (analise por sujeito do pipeline principal)
TARGET_SUBJECTS = ['f02', 'f03', 'f04', 'f18']

RF_PARAMS = dict(n_estimators=250, max_features=0.6, random_state=42, n_jobs=-1)

# Features-chave para a delta (subject-adaptive)
KEY_FEATURES = ['hr_mean', 'rmssd', 'pnn50', 'mean_raw_eda', 'mean_tonic_eda',
                'bvp_std', 'sdnn', 'HF_n', 'ratio', 'acc_std', 'peaks_density']

# Razoes (subject-independent).
RATIO_SPECS = [
    ('cv_eda',           'std_tonic_eda',  'mean_tonic_eda', True),
    ('cv_hr',            'hr_std',         'hr_mean',        False),
    ('rmssd_sdnn_ratio', 'rmssd',          'sdnn',           False),
    ('hf_lf_ratio',      'HF_power',       'LF_power',       False),
    ('pnn_ratio',        'pnn50',          'pnn20',          False),
    ('bvp_hr_ratio',     'bvp_std',        'hr_mean',        False),
    ('phasic_raw_ratio', 'std_phasic_eda', 'std_raw_eda',    False),
    ('cv_acc',           'acc_std',        'acc_mean',       False),
]

CALIB_FRAC = 0.5
SEED = 42



def add_ratio_features(df):
    """Razoes e coeficientes de variacao, por janela. Subject-independent:
    nao usam baseline por sujeito. Constroi apenas as que tem colunas-fonte."""
    df_r = df.copy()
    built = []
    for name, num, den, use_abs in RATIO_SPECS:
        if num in df.columns and den in df.columns:
            denom = df[den].abs() if use_abs else df[den]
            df_r[name] = df[num] / (denom + 1e-6)
            built.append(name)
    return df_r, built


def build_calibrated_delta(train_df, test_df, key_features,
                           calib_frac=CALIB_FRAC, seed=SEED):
    """Delta em relacao ao baseline REST do proprio sujeito (subject-adaptive).

    Treino: baseline de cada sujeito de treino vem do REST dele.
    Teste: para cada sujeito de teste, separa uma fracao das janelas REST como
    calibracao (enrollment), estima o baseline e calcula a delta para as demais
    janelas. As janelas de calibracao NAO entram na avaliacao, evitando
    circularidade. Retorna treino e teste com deltas, a lista de colunas delta
    e os indices de avaliacao do teste.
    """
    rng = np.random.RandomState(seed)
    tr = train_df.copy()
    te = test_df.copy()
    delta_cols = [f'delta_{f}' for f in key_features]

    #TREINO: baseline = REST
    for subj in tr['subject_id'].unique():
        rest = tr[(tr['subject_id'] == subj) & (tr[TARGET] == 0)][key_features]
        base = rest.mean() if len(rest) else \
            tr[tr['subject_id'] == subj][key_features].mean()
        m = tr['subject_id'] == subj
        for f in key_features:
            denom = abs(base[f]) if base[f] != 0 else 1.0
            tr.loc[m, f'delta_{f}'] = (tr.loc[m, f] - base[f]) / (denom + 1e-12)

    #TESTE: calibracao com parte do REST do proprio sujeito
    eval_idx = []
    for subj in te['subject_id'].unique():
        sub = te[te['subject_id'] == subj]
        rest_idx = sub[sub[TARGET] == 0].index.tolist()
        rng.shuffle(rest_idx)
        n_calib = max(1, int(len(rest_idx) * calib_frac))
        calib_idx = set(rest_idx[:n_calib])

        base = te.loc[list(calib_idx), key_features].mean()
        m = te['subject_id'] == subj
        for f in key_features:
            denom = abs(base[f]) if base[f] != 0 else 1.0
            te.loc[m, f'delta_{f}'] = (te.loc[m, f] - base[f]) / (denom + 1e-12)

        eval_idx.extend([i for i in sub.index if i not in calib_idx])

    return tr, te, delta_cols, sorted(eval_idx)


def metrics_block(y_true, y_pred, y_prob=None):
    out = {
        'f1': f1_score(y_true, y_pred),
        'prec': precision_score(y_true, y_pred, zero_division=0),
        'rec': recall_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None and len(np.unique(y_true)) > 1:
        out['auc'] = roc_auc_score(y_true, y_prob)
    else:
        out['auc'] = np.nan
    return out


def subj_metrics(yt, yp, yprob):
    auc = roc_auc_score(yt, yprob) if len(np.unique(yt)) > 1 else np.nan
    return auc, recall_score(yt, yp, zero_division=0)


def best_threshold(yt, yprob):
    """Limiar que maximiza o F1 do sujeito. Retorna limiar, F1 e recall nesse
    ponto. Um recall baixo em 0.5 que sobe muito no limiar otimo indica caso
    separavel, porem silenciado pelo ponto de operacao (problema de limiar)."""
    grid = np.linspace(0.10, 0.90, 81)
    best_t, best_f1, best_rec = 0.5, -1.0, 0.0
    for t in grid:
        yp = (yprob >= t).astype(int)
        f1 = f1_score(yt, yp, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
            best_rec = recall_score(yt, yp, zero_division=0)
    return best_t, best_f1, best_rec



# %% LOOP SOBRE OS 10 FOLDS

data = pd.read_csv(DATA_PATH)
gkf = GroupKFold(n_splits=N_FOLDS)
fold_splits = list(gkf.split(data, data[TARGET], data['subject_id']))
print(f"Dataset: {data.shape} | sujeitos: {data['subject_id'].nunique()} | "
      f"folds: {N_FOLDS}")

rec_A = []
rec_B = []
subj_A = []
subj_B = []
delta_check = []  

for fold, (tr_idx, te_idx) in enumerate(fold_splits, start=1):
    train = data.iloc[tr_idx].reset_index(drop=True)
    test = data.iloc[te_idx].reset_index(drop=True)

    feature_cols = [c for c in test.columns if c not in COLS_TO_DROP]
    key_feats = [f for f in KEY_FEATURES if f in feature_cols]
    y_train = train[TARGET]
    y_test = test[TARGET]

    # ---------- PARADIGMA A: subject-independent ----------
    # Baseline absoluto
    m_base = RandomForestClassifier(**RF_PARAMS)
    m_base.fit(train[feature_cols], y_train)
    p_base = m_base.predict(test[feature_cols])
    pr_base = m_base.predict_proba(test[feature_cols])[:, 1]
    mb = metrics_block(y_test, p_base, pr_base)

    # Absoluto + razoes
    train_r, built = add_ratio_features(train)
    test_r, _ = add_ratio_features(test)
    feat_r = feature_cols + built
    m_ratio = RandomForestClassifier(**RF_PARAMS)
    m_ratio.fit(train_r[feat_r], y_train)
    p_ratio = m_ratio.predict(test_r[feat_r])
    pr_ratio = m_ratio.predict_proba(test_r[feat_r])[:, 1]
    mr = metrics_block(y_test, p_ratio, pr_ratio)

    rec_A.append({'fold': fold,
                  'base_f1': mb['f1'], 'base_auc': mb['auc'],
                  'ratio_f1': mr['f1'], 'ratio_auc': mr['auc'],
                  'd_f1': mr['f1'] - mb['f1'], 'd_auc': mr['auc'] - mb['auc']})

    # por sujeito-alvo no paradigma A: baseline vs razoes, mais limiar otimo
    for s in TARGET_SUBJECTS:
        mask = (test['subject_id'] == s).values
        if mask.sum() == 0:
            continue
        yt = y_test[mask].values
        auc_b, rec_b = subj_metrics(yt, p_base[mask], pr_base[mask])
        auc_r, rec_r = subj_metrics(yt, p_ratio[mask], pr_ratio[mask])
        t_opt, f1_opt, rec_opt = best_threshold(yt, pr_ratio[mask])
        subj_A.append({'subject': s, 'fold': fold, 'n': int(mask.sum()),
                       'auc_base': auc_b, 'auc_ratio': auc_r,
                       'rec_0.5': rec_r, 'thr_opt': t_opt,
                       'rec_thr_opt': rec_opt})

    # PARADIGMA B: subject-adaptive (delta calibrada)
    tr_d, te_d, delta_cols, eval_idx = build_calibrated_delta(
        train, test, key_feats)

    # deltas não podem estar zeradas no teste 
    dsum = te_d.loc[eval_idx, delta_cols].abs().to_numpy().sum()
    delta_check.append({'fold': fold, 'delta_abs_sum_test': dsum})

    yev = te_d.loc[eval_idx, TARGET]

    # baseline absoluto avaliado nas mesmas janelas de eval
    pb_ev = m_base.predict(te_d.loc[eval_idx, feature_cols])
    prb_ev = m_base.predict_proba(te_d.loc[eval_idx, feature_cols])[:, 1]
    mbe = metrics_block(yev, pb_ev, prb_ev)

    # absoluto + delta calibrada
    feat_d = feature_cols + delta_cols
    m_delta = RandomForestClassifier(**RF_PARAMS)
    m_delta.fit(tr_d[feat_d].fillna(0), y_train)
    pd_ev = m_delta.predict(te_d.loc[eval_idx, feat_d].fillna(0))
    prd_ev = m_delta.predict_proba(te_d.loc[eval_idx, feat_d].fillna(0))[:, 1]
    mde = metrics_block(yev, pd_ev, prd_ev)

    rec_B.append({'fold': fold,
                  'base_f1': mbe['f1'], 'base_auc': mbe['auc'],
                  'delta_f1': mde['f1'], 'delta_auc': mde['auc'],
                  'd_f1': mde['f1'] - mbe['f1'], 'd_auc': mde['auc'] - mbe['auc']})

    # por sujeito-alvo no paradigma B: baseline-eval vs delta calibrada
    eval_subj = te_d.loc[eval_idx, 'subject_id'].values
    yev_arr = yev.values
    for s in TARGET_SUBJECTS:
        m = (eval_subj == s)
        if m.sum() == 0:
            continue
        yt = yev_arr[m]
        auc_be, _ = subj_metrics(yt, pb_ev[m], prb_ev[m])
        auc_de, rec_de = subj_metrics(yt, pd_ev[m], prd_ev[m])
        d_auc = (auc_de - auc_be) if not (np.isnan(auc_be) or np.isnan(auc_de)) \
            else np.nan
        subj_B.append({'subject': s, 'fold': fold, 'n_eval': int(m.sum()),
                       'auc_base_eval': auc_be, 'auc_delta': auc_de,
                       'd_auc': d_auc, 'rec_delta': rec_de})

    print(f"  Fold {fold:2d} | A ratios dF1={rec_A[-1]['d_f1']:+.3f} "
          f"dAUC={rec_A[-1]['d_auc']:+.3f} | B delta dF1={rec_B[-1]['d_f1']:+.3f} "
          f"dAUC={rec_B[-1]['d_auc']:+.3f} | delta_test_sum={dsum:.1f}")

A = pd.DataFrame(rec_A)
B = pd.DataFrame(rec_B)
chk = pd.DataFrame(delta_check)
SA = pd.DataFrame(subj_A)
SB = pd.DataFrame(subj_B)


# %% VERIFICACAO DAS DELTAS NO TESTE
print(chk.round(2).to_string(index=False))
if (chk['delta_abs_sum_test'] > 0).all():
    print("paradigma adaptive correto")
else:
    print("Tem algum delta zerado")


# %% RESUMO PARADIGMA A (subject-independent)
print("PARADIGMA A - SUBJECT-INDEPENDENT (razoes, sem baseline por sujeito)")

print(f"  Baseline : F1={A.base_f1.mean():.4f}+/-{A.base_f1.std():.4f} | "
      f"AUC={A.base_auc.mean():.4f}")
print(f"  + Razoes : F1={A.ratio_f1.mean():.4f}+/-{A.ratio_f1.std():.4f} | "
      f"AUC={A.ratio_auc.mean():.4f}")
print(f"  dF1 medio={A.d_f1.mean():+.4f} | dAUC medio={A.d_auc.mean():+.4f} | "
      f"folds com melhora F1: {(A.d_f1 > 0).sum()}/10")
st, p = stats.wilcoxon(A.ratio_f1, A.base_f1)
print(f"  Wilcoxon F1  (razoes vs baseline): p={p:.4f} "
      f"-> {'significativo' if p < 0.05 else 'nao significativo'}")
st, pa = stats.wilcoxon(A.ratio_auc, A.base_auc)
print(f"  Wilcoxon AUC (razoes vs baseline): p={pa:.4f} "
      f"-> {'significativo' if pa < 0.05 else 'nao significativo'}")

# %% RESUMO PARADIGMA B (subject-adaptive)
print("PARADIGMA B - SUBJECT-ADAPTIVE (delta calibrada com REST do sujeito)")
print(f"  Baseline (eval): F1={B.base_f1.mean():.4f} | AUC={B.base_auc.mean():.4f}")
print(f"  + Delta calib  : F1={B.delta_f1.mean():.4f} | AUC={B.delta_auc.mean():.4f}")
print(f"  dF1 medio={B.d_f1.mean():+.4f} | dAUC medio={B.d_auc.mean():+.4f} | "
      f"folds com melhora F1: {(B.d_f1 > 0).sum()}/10")
st, p = stats.wilcoxon(B.delta_f1, B.base_f1)
print(f"  Wilcoxon F1  (delta vs baseline): p={p:.4f} "
      f"-> {'significativo' if p < 0.05 else 'nao significativo'}")
st, pb = stats.wilcoxon(B.delta_auc, B.base_auc)
print(f"  Wilcoxon AUC (delta vs baseline): p={pb:.4f} "
      f"-> {'significativo' if pb < 0.05 else 'nao significativo'}")

# %% SUJEITOS DISSOCIADOS - PARADIGMA A (baseline vs razoes + limiar)

print("SUJEITOS DISSOCIADOS - PARADIGMA A (baseline vs razoes, e limiar otimo)")
if not SA.empty:
    print(SA.round(3).to_string(index=False))

# %% SUJEITOS DISSOCIADOS - PARADIGMA B (delta calibrada)
print("SUJEITOS DISSOCIADOS - PARADIGMA B (baseline-eval vs delta calibrada)")
if not SB.empty:
    print(SB.round(3).to_string(index=False))

