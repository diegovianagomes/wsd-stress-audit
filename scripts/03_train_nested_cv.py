# %% 1. Setup
""" 1. Setup
Nested Cross-Validation com GroupKFold. Cenario 1: REST vs STRESS.

Principio metodologico central: nenhuma etapa que aprende parametros (escala,
selecao de features, resampling, hiperparametros) enxerga dados de teste. Tudo
e ajustado apenas no treino de cada fold, encapsulado em Pipeline. Isso evita
as tres fontes de vazamento comuns na literatura de referencia: 
 - CV sem agrupamento por sujeito, 
 - Escala global,
 - Selecao de features global.

Objetivo do estudo: ir alem da metrica agregada e responder quem o modelo erra
e por que. Por isso o pipeline salva a predicao de cada janela no fold externo
e analisa o erro por sujeito.

"""

import io
import time
import pickle
import itertools
from pathlib import Path
from datetime import datetime
from functools import partial

import pandas as pd
import numpy as np

from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    roc_auc_score, precision_recall_curve, auc
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import (
    RandomOverSampler, SMOTE, BorderlineSMOTE, SVMSMOTE, ADASYN
)
from imblearn.under_sampling import (
    RandomUnderSampler, TomekLinks, EditedNearestNeighbours, NearMiss,
    ClusterCentroids, NeighbourhoodCleaningRule, OneSidedSelection
)
from imblearn.combine import SMOTEENN, SMOTETomek

import sys
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_DIR, EXPERIMENTS_DIR, RANDOM_SEED


# %% 2. Configuracao
"""
Caminhos, grids de hiperparametros e parametros do experimento.
"""

# Caminhos
RESULTS_DIR = EXPERIMENTS_DIR / "results" / "nested_cv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Dados
CSV_PATH = PROCESSED_DIR / "dataset_stress.csv"
TARGET = "label"
GROUP_COL = "subject_id"
SAMPLE_COL = "window_id"
META_COLS = ['subject_id', 'window_id', 'label', 'protocol', 'scenario']

# Nested CV
N_OUTER_FOLDS = 10
N_INNER_FOLDS = 10
METRIC_SORT = "f1"

# Listas de modelos
CLASSIC_MODELS = ['KNN', 'LogReg', 'GaussianNB', 'SVC']
TREE_MODELS = ['RF', 'XGB', 'LGBM']
MODELS = CLASSIC_MODELS + TREE_MODELS
SCALE_SENSITIVE = {'KNN', 'LogReg', 'SVC'}

SELECT_SCORER = partial(mutual_info_classif, random_state=RANDOM_SEED)

# Parametros inteiros a converter 
INT_PARAMS = ['n_estimators', 'max_depth', 'num_leaves',
              'min_samples_leaf', 'n_neighbors']

# Grids de hiperparametros
PARAM_GRIDS = {
    'RF': {
        'n_estimators': [150, 200, 250],
        'max_features': [0.6, 0.7, 0.8],
        'min_samples_leaf': [1, 2, 3],
    },
    'XGB': {
        'n_estimators': [200, 300],
        'learning_rate': [0.08, 0.1, 0.12],
        'max_depth': [8, 10],
        'subsample': [0.9, 1.0],
    },
    'LGBM': {
        'n_estimators': [200, 300],
        'learning_rate': [0.08, 0.1, 0.12],
        'max_depth': [8, 10],
        'num_leaves': [31, 50],
    },
    'KNN': {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'p': [1, 2],
    },
    'LogReg': {
        'C': [0.01, 0.1, 1, 10, 100],
        'penalty': ['l2'],
        'solver': ['lbfgs'],
    },
    'GaussianNB': {
        'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6],
    },
    'SVC': {
        'C': [0.1, 1, 10],
        'kernel': ['rbf'],
        'gamma': ['scale', 'auto'],
    },
}

print(f"Dataset: {CSV_PATH}")
print(f"Outer folds: {N_OUTER_FOLDS} | Inner folds: {N_INNER_FOLDS}")
for model_name in MODELS:
    combos = 1
    for v in PARAM_GRIDS[model_name].values():
        combos *= len(v)
    print(f"  {model_name}: {combos} combinacoes")


# %% 3. Catalogo de tecnicas de resampling
""" 3. Catalogo de resampling
Tecnicas do imbalanced-learn. Cada entrada e uma factory (lambda) que devolve
instancia nova, garantindo estado limpo a cada fold. O Pipeline do imblearn
(em create_model) assegura que o resampling so atua sobre o treino. Definido
antes das funcoes porque create_model o referencia.

"""

RESAMPLERS = {
    'none': None,
    # Oversampling
    'RandomOver':       lambda: RandomOverSampler(random_state=RANDOM_SEED),
    'SMOTE':            lambda: SMOTE(random_state=RANDOM_SEED),
    'BorderlineSMOTE':  lambda: BorderlineSMOTE(random_state=RANDOM_SEED),
    'SVMSMOTE':         lambda: SVMSMOTE(random_state=RANDOM_SEED),
    'ADASYN':           lambda: ADASYN(random_state=RANDOM_SEED),
    # Undersampling
    'RandomUnder':      lambda: RandomUnderSampler(random_state=RANDOM_SEED),
    'TomekLinks':       lambda: TomekLinks(),
    'ENN':              lambda: EditedNearestNeighbours(),
    'NearMiss1':        lambda: NearMiss(version=1),
    'NearMiss2':        lambda: NearMiss(version=2),
    'NearMiss3':        lambda: NearMiss(version=3),
    'ClusterCentroids': lambda: ClusterCentroids(random_state=RANDOM_SEED),
    'NCR':              lambda: NeighbourhoodCleaningRule(),
    'OneSidedSel':      lambda: OneSidedSelection(random_state=RANDOM_SEED),
    # Combinadas
    'SMOTETomek':       lambda: SMOTETomek(random_state=RANDOM_SEED),
    'SMOTEENN':         lambda: SMOTEENN(random_state=RANDOM_SEED),
}

print(f"Tecnicas de resampling: {len(RESAMPLERS)}")
print(", ".join(RESAMPLERS.keys()))


# %% 4. Funcoes auxiliares
""" 4. Funcoes auxiliares
    create_model         : instancia o modelo (selecao/scaler/resampler opc.).
    evaluate_predictions : metricas de classificacao binaria.
    normalize_by_subject : z-score intra-sujeito (sem leakage entre sujeitos).
    run_nested_cv        : nested CV; salva predicoes por janela se solicitado.
    summarize            : resumo por modelo.
    summarize_resampling : ranking por tecnica de resampling.
"""


def create_model(model_name, params, resampler_name=None, select_k=None):
    """Instancia o modelo, opcionalmente dentro de um Pipeline.

    Ordem dos passos: select -> scaler -> resampler -> clf. Cada passo opcional
    so entra quando solicitado. Todos sao ajustados apenas no fit (treino do
    fold): selecao, escala e resampling nunca enxergam o teste. O Pipeline do
    imblearn ignora o resampler no predict.
    """
    if model_name == 'RF':
        clf = RandomForestClassifier(
            n_jobs=-1, random_state=RANDOM_SEED, **params
        )
    elif model_name == 'XGB':
        clf = XGBClassifier(
            n_jobs=-1, random_state=RANDOM_SEED,
            eval_metric='logloss', verbosity=0, **params
        )
    elif model_name == 'LGBM':
        clf = LGBMClassifier(
            n_jobs=-1, random_state=RANDOM_SEED, verbose=-1, **params
        )
    elif model_name == 'KNN':
        clf = KNeighborsClassifier(n_jobs=-1, **params)
    elif model_name == 'LogReg':
        clf = LogisticRegression(
            random_state=RANDOM_SEED, max_iter=1000, **params
        )
    elif model_name == 'GaussianNB':
        clf = GaussianNB(**params)
    elif model_name == 'SVC':
        clf = SVC(random_state=RANDOM_SEED, probability=True, **params)
    else:
        raise ValueError(f"Modelo desconhecido: {model_name}")

    steps = []
    if select_k is not None:
        steps.append(('select', SelectKBest(SELECT_SCORER, k=select_k)))
    if model_name in SCALE_SENSITIVE:
        steps.append(('scaler', StandardScaler()))
    if resampler_name not in (None, 'none'):
        steps.append(('resampler', RESAMPLERS[resampler_name]()))
    steps.append(('clf', clf))

    if len(steps) == 1:
        return clf
    return ImbPipeline(steps)


def evaluate_predictions(y_true, y_pred, y_proba):
    """Calcula metricas de classificacao binaria."""
    prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_proba)
    return {
        'f1': f1_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_proba),
        'auprc': auc(rec_curve, prec_curve),
    }


def normalize_by_subject(df, feat_cols, group_col=GROUP_COL):
    """Z-score intra-sujeito para cada feature.

    Cada sujeito usa apenas os proprios dados (sem leakage entre sujeitos nem
    de rotulo). E uma normalizacao transdutiva: usa a sessao completa do
    sujeito de teste, o que deve ser declarado como personalizacao.
    """
    df_norm = df.copy()
    for subj, idx in df.groupby(group_col).groups.items():
        subj_data = df.loc[idx, feat_cols]
        means = subj_data.mean()
        stds = subj_data.std().replace(0, 1)  # evita divisao por zero
        df_norm.loc[idx, feat_cols] = (subj_data - means) / stds
    return df_norm


def run_nested_cv(X, y, groups, models=MODELS, cv_class=GroupKFold,
                  cv_kwargs=None, extra_fields=None, track_resources=False,
                  resampler_name=None, select_k=None,
                  sample_ids=None, predictions_path=None, label=""):
    
    """Executa nested CV e retorna um DataFrame de metricas por (modelo, fold).

    resampler_name  : tecnica em RESAMPLERS (ou None), aplicada so ao treino.
    select_k        : nº de features (SelectKBest intra-fold) ou None (todas).
    sample_ids      : Serie alinhada a X com o id da janela (window_id).
    predictions_path: se dado, salva a predicao de cada janela do fold externo
                      (model, fold, subject, sample_id, y_true, y_pred, y_prob).
    """
    cv_kwargs = cv_kwargs or {}
    extra_fields = extra_fields or {}
    results = []
    pred_frames = []

    for model_name in models:
        param_grid = PARAM_GRIDS[model_name]
        param_names = list(param_grid.keys())
        param_combos = list(itertools.product(*param_grid.values()))

        print(f"\n{'='*60}")
        print(f"MODELO: {model_name} {label}".rstrip())
        print(f"{'='*60}")

        outer_cv = cv_class(n_splits=N_OUTER_FOLDS, **cv_kwargs)

        for fold_outer, (train_idx, test_idx) in enumerate(
            outer_cv.split(X, y, groups), start=1
        ):
            X_train_outer, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train_outer, y_test = y.iloc[train_idx], y.iloc[test_idx]
            groups_train_outer = groups.iloc[train_idx]
            test_subjects = groups.iloc[test_idx].unique().tolist()

            # --- LOOP INTERNO: grid search ---
            inner_cv = cv_class(n_splits=N_INNER_FOLDS, **cv_kwargs)
            best_score = -1
            best_params = None

            for combo in param_combos:
                params = dict(zip(param_names, combo))
                for k in INT_PARAMS:
                    if k in params:
                        params[k] = int(params[k])

                inner_scores = []
                for train_inner, val_inner in inner_cv.split(
                    X_train_outer, y_train_outer, groups_train_outer
                ):
                    model = create_model(model_name, params,
                                         resampler_name, select_k)
                    model.fit(
                        X_train_outer.iloc[train_inner],
                        y_train_outer.iloc[train_inner],
                    )
                    y_pred_val = model.predict(X_train_outer.iloc[val_inner])
                    inner_scores.append(
                        f1_score(y_train_outer.iloc[val_inner], y_pred_val)
                    )

                mean_score = np.mean(inner_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_params = params.copy()

            # --- AVALIACAO EXTERNA com o melhor hiperparametro ---
            model_final = create_model(model_name, best_params,
                                       resampler_name, select_k)

            t0 = time.time()
            model_final.fit(X_train_outer, y_train_outer)
            train_time = time.time() - t0

            t0 = time.time()
            y_pred = model_final.predict(X_test)
            y_proba = model_final.predict_proba(X_test)[:, 1]
            pred_time = time.time() - t0

            metrics = evaluate_predictions(y_test, y_pred, y_proba)

            # --- PREDICOES POR JANELA (para analise por sujeito) ---
            if predictions_path is not None:
                if sample_ids is not None:
                    sid = sample_ids.iloc[test_idx].values
                else:
                    sid = X_test.index.values
                pred_frames.append(pd.DataFrame({
                    'model': model_name,
                    'resampler': resampler_name or 'none',
                    'select_k': 'all' if select_k is None else select_k,
                    'fold': fold_outer,
                    'subject': groups.iloc[test_idx].values,
                    'sample_id': sid,
                    'y_true': y_test.values,
                    'y_pred': y_pred,
                    'y_prob': y_proba,
                }))

            row = {
                'model': model_name,
                'resampler': resampler_name or 'none',
                'select_k': 'all' if select_k is None else select_k,
                'fold': fold_outer,
                'test_subjects': str(test_subjects),
                'best_params': str(best_params),
                'inner_best_f1': best_score,
                **extra_fields,
                **metrics,
            }
            if track_resources:
                buffer = io.BytesIO()
                pickle.dump(model_final, buffer)
                row['train_time_s'] = train_time
                row['pred_time_s'] = pred_time
                row['model_size_bytes'] = buffer.tell()

            results.append(row)
            print(f"  Fold {fold_outer:2d} | F1={metrics['f1']:.3f} | "
                  f"AUC={metrics['roc_auc']:.3f} | Subjects: {test_subjects}")

    if predictions_path is not None and pred_frames:
        pd.concat(pred_frames, ignore_index=True).to_csv(
            predictions_path, index=False)
        print(f"\nPredicoes salvas em {predictions_path}")

    return pd.DataFrame(results)


def summarize(df, models=MODELS, title=None):
    cols = ['f1', 'precision', 'recall', 'roc_auc', 'auprc']
    if title:
        print(f"\n=== {title} ===\n")
    for m in models:
        row = df.query("model == @m")[cols]
        if row.empty:
            continue
        print(f"{m}: F1={row['f1'].mean():.3f}+/-{row['f1'].std():.3f} | "
              f"AUC={row['roc_auc'].mean():.3f} | "
              f"Recall={row['recall'].mean():.3f}")
    best_model = df.groupby('model')['f1'].mean().idxmax()
    best_f1 = df.groupby('model')['f1'].mean().max()
    print(f"\nMelhor modelo: {best_model} (F1 medio = {best_f1:.4f})")


def summarize_resampling(df, models=None, sort_metric='f1'):
    models = models or df['model'].unique().tolist()
    cols = ['f1', 'precision', 'recall', 'roc_auc', 'auprc']
    best_by_model = {}
    for m in models:
        sub = df[df['model'] == m]
        if sub.empty:
            continue
        agg = (sub.groupby('resampler')[cols]
                  .agg(['mean', 'std'])
                  .sort_values((sort_metric, 'mean'), ascending=False))
        print(f"\n{m} (ordenado por {sort_metric})")
        for rs, r in agg.iterrows():
            print(f"{rs:16s} "
                  f" F1:{r[('f1', 'mean')]:.3f}+/-{r[('f1', 'std')]:.3f} | "
                  f"  P:{r[('precision', 'mean')]:.3f} | "
                  f"  R:{r[('recall', 'mean')]:.3f} | "
                  f"AUC:{r[('roc_auc', 'mean')]:.3f}")
        best = agg.index[0]
        best_by_model[m] = best
        print(f"  melhor tecnica ({m}): {best}")
    return best_by_model


# %% 5. Carregamento dos dados (Cenario 1: REST vs STRESS)
"""
Define X, y, groups, sample_ids e feat_cols, usados pelas celulas seguintes.
"""

data = pd.read_csv(CSV_PATH)
feat_cols = [c for c in data.columns if c not in META_COLS]
X = data[feat_cols]
y = data[TARGET]
groups = data[GROUP_COL]
sample_ids = data[SAMPLE_COL]

print(f"Shape: {X.shape} | Sujeitos: {groups.nunique()}")
print(f"Labels: {y.value_counts().to_dict()}")


# %% 6. Experimento A: baseline sem resampling, 49 features (referencia)
""" 6. Baseline
Nested CV com features absolutas, sem resampling nem selecao, regime completo
(N_INNER_FOLDS = 10). SVC com probability=True e lento; use TREE_MODELS.
"""

N_INNER_FOLDS = 10
EXP_A_MODELS = TREE_MODELS

df_baseline = run_nested_cv(
    X, y, groups,
    models=EXP_A_MODELS,
    extra_fields={'strategy': 'absolute', 'n_features': len(feat_cols)},
    track_resources=True,
    label="(BASELINE 49)",
)
df_baseline.to_csv(RESULTS_DIR / "nested_cv_baseline.csv", index=False)
summarize(df_baseline, models=EXP_A_MODELS, title="BASELINE - 49 features")


# %% 7. Experimento B: varredura de resampling nas arvores 
"""
Arvores contra none e RandomUnder. Loop interno em 5 folds (com 36 sujeitos, 5
internos sao mais estaveis que 10 e custam metade; verificou-se que 5 e 10
diferem em no maximo 0.008 de AUC). Salva a cada par e retoma.
"""

N_INNER_FOLDS = 5

SWEEP_MODELS = TREE_MODELS
SWEEP_RESAMPLERS = ['none', 'RandomUnder']
OUT_PATH = RESULTS_DIR / "nested_cv_resampling_trees_5inner.csv"

if OUT_PATH.exists():
    df_done = pd.read_csv(OUT_PATH)
    done_pairs = set(zip(df_done['model'], df_done['resampler']))
    print(f"Retomando: {len(done_pairs)} pares prontos.")
else:
    df_done = pd.DataFrame()
    done_pairs = set()
    print("Iniciando do zero.")

combos = [(m, rs) for m in SWEEP_MODELS for rs in SWEEP_RESAMPLERS]
todo = [(m, rs) for (m, rs) in combos if (m, rs) not in done_pairs]
print(f"Inner folds: {N_INNER_FOLDS} | Total: {len(combos)} | "
      f"Concluidos: {len(done_pairs)} | Restantes: {len(todo)}")

t_start = time.time()
failures = []

for i, (model_name, rs_name) in enumerate(todo, start=1):
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'#'*60}")
    print(f"[{stamp}] ({i}/{len(todo)}) MODELO={model_name} RESAMPLER={rs_name}")
    print(f"{'#'*60}")
    t0 = time.time()
    try:
        df_rs = run_nested_cv(
            X, y, groups,
            models=[model_name],
            resampler_name=rs_name,
            extra_fields={'strategy': 'absolute', 'n_features': len(feat_cols),
                          'n_inner': N_INNER_FOLDS},
            track_resources=True,
            label=f"[{rs_name}]",
        )
        df_done = pd.concat([df_done, df_rs], ignore_index=True)
        df_done.to_csv(OUT_PATH, index=False)
        dt = (time.time() - t0) / 60
        print(f"  OK em {dt:.1f} min | salvo ({len(df_done)} linhas) "
              f"em {OUT_PATH.name}")
    except Exception as e:
        dt = (time.time() - t0) / 60
        failures.append((model_name, rs_name, f"{type(e).__name__}: {e}"))
        print(f"  [FALHA apos {dt:.1f} min] {model_name}/{rs_name}: "
              f"{type(e).__name__}: {e}")

total_min = (time.time() - t_start) / 60
print(f"\n{'='*60}")
print(f"VARREDURA CONCLUIDA em {total_min:.1f} min")
print(f"Linhas totais: {len(df_done)} | Falhas: {len(failures)}")
for m, rs, err in failures:
    print(f"  FALHA {m}/{rs}: {err}")
print(f"{'='*60}")


# %% 8. Ranking das tecnicas de resampling
"""
Ordena as tecnicas por F1, com AUC ao lado, pois o F1 num limiar fixo pode
premiar deslocamento de ponto de operacao sem ganho de discriminacao.
"""

df_resampling = pd.read_csv(OUT_PATH)
best_by_model = summarize_resampling(df_resampling)


# %% 9. Experimento C: selecao de features 49 vs 13 (selecao intra-fold)
""" 
Replica a reducao do artigo (49 -> 13 para estresse binario), mas com a selecao
refeita dentro do treino de cada fold (a selecao global do artigo vaza o teste).
SelectKBest por informacao mutua como primeiro passo do Pipeline. 5 internos.
"""

N_INNER_FOLDS = 5

SELECT_MODELS = TREE_MODELS
SELECT_KS = [None, 13]
OUT_PATH_FS = RESULTS_DIR / "nested_cv_featsel_trees.csv"

if OUT_PATH_FS.exists():
    df_fs = pd.read_csv(OUT_PATH_FS)
    done_fs = set(zip(df_fs['model'], df_fs['select_k'].astype(str)))
    print(f"Retomando selecao: {len(done_fs)} pares prontos.")
else:
    df_fs = pd.DataFrame()
    done_fs = set()
    print("Iniciando selecao do zero.")

fs_combos = [(m, k) for m in SELECT_MODELS for k in SELECT_KS]
fs_todo = [(m, k) for (m, k) in fs_combos
           if (m, 'all' if k is None else str(k)) not in done_fs]
print(f"Inner folds: {N_INNER_FOLDS} | Restantes: {len(fs_todo)}")

for i, (model_name, k) in enumerate(fs_todo, start=1):
    k_lbl = 'all' if k is None else str(k)
    print(f"\n{'#'*60}")
    print(f"({i}/{len(fs_todo)}) MODELO={model_name} | features={k_lbl}")
    print(f"{'#'*60}")
    try:
        df_k = run_nested_cv(
            X, y, groups,
            models=[model_name],
            resampler_name='none',
            select_k=k,
            extra_fields={'strategy': 'absolute',
                          'n_features': len(feat_cols) if k is None else k,
                          'n_inner': N_INNER_FOLDS},
            track_resources=True,
            label=f"[features={k_lbl}]",
        )
        df_fs = pd.concat([df_fs, df_k], ignore_index=True)
        df_fs.to_csv(OUT_PATH_FS, index=False)
        print(f"  OK | salvo ({len(df_fs)} linhas) em {OUT_PATH_FS.name}")
    except Exception as e:
        print(f"  [FALHA] {model_name}/{k_lbl}: {type(e).__name__}: {e}")

print("\nSELECAO DE FEATURES: 49 ou 13 (media +/- desvio)")
cols_fs = ['f1', 'precision', 'recall', 'roc_auc', 'auprc']
for m in SELECT_MODELS:
    for k_lbl in ['all', '13']:
        r = df_fs[(df_fs.model == m) &
                  (df_fs['select_k'].astype(str) == k_lbl)][cols_fs]
        if r.empty:
            continue
        tag = '49' if k_lbl == 'all' else '13'
        print(f"  {m:5s} [{tag:>2s}] F1={r.f1.mean():.3f}+/-{r.f1.std():.3f} | "
              f"AUC={r.roc_auc.mean():.3f} | AUPRC={r.auprc.mean():.3f}")


# %% 10. Inspecao descritiva das features selecionadas
"""
Ranking de informacao mutua sobre o todo do conjunto, apenas para visualizar quais
features dominam e analisar estabilidade. não e usado na avaliação, onde a
selecao e refeita dentro de cada fold de treino.
"""

sel_desc = SelectKBest(SELECT_SCORER, k=13).fit(X, y)
ranking_mi = pd.Series(sel_desc.scores_, index=feat_cols).sort_values(
    ascending=False)
selected_13 = ranking_mi.head(13).index.tolist()

print("Melhores features:")
for f, s in ranking_mi.head(13).items():
    print(f"  {f:30s} MI={s:.4f}")


# %% 11. Comparacao final
""" 
Nested CV no regime completo (N_INNER_FOLDS = 10) com a configuracao escolhida.
Nenhuma tecnica supera o baseline de forma relevante; 'none' e a referencia
honesta. SELECT_K_FINAL = None usa as 49 features; 13 replica o artigo.
"""

N_INNER_FOLDS = 10
BEST_RESAMPLER = 'none'
SELECT_K_FINAL = None
FINAL_MODELS = TREE_MODELS

tag_k = 'all' if SELECT_K_FINAL is None else str(SELECT_K_FINAL)
df_final = run_nested_cv(
    X, y, groups,
    models=FINAL_MODELS,
    resampler_name=BEST_RESAMPLER,
    select_k=SELECT_K_FINAL,
    extra_fields={'strategy': 'absolute', 'resampling': BEST_RESAMPLER,
                  'n_inner': N_INNER_FOLDS},
    track_resources=True,
    label=f"[{BEST_RESAMPLER} | k={tag_k}]",
)
df_final.to_csv(
    RESULTS_DIR / f"nested_cv_final_{BEST_RESAMPLER}_k{tag_k}.csv", index=False)
summarize(df_final, models=FINAL_MODELS,
          title=f"FINAL - {BEST_RESAMPLER} | {tag_k} features (10 inner)")


# %% 12. Predicoes por janela
"""
Salva [model, fold, subject, sample_id, y_true, y_pred, y_prob] de cada janela
do fold externo, no regime honesto (49 features, sem resampling). 5 internos.
Base para a analise por sujeito da celula 14.
"""

N_INNER_FOLDS = 5
PRED_PATH = RESULTS_DIR / "predictions_nested_cv.csv"

_ = run_nested_cv(
    X, y, groups,
    models=TREE_MODELS,
    resampler_name='none',
    select_k=None,
    sample_ids=sample_ids,
    predictions_path=PRED_PATH,
    extra_fields={'strategy': 'absolute', 'n_inner': N_INNER_FOLDS},
    label="(PRED)",
)


# %% 13. Analise de erro por sujeito
"""
A partir das predicoes, calcula por sujeito: acuracia, recall de estresse, AUC
intra-sujeito e probabilidade media. Ordena pelos piores. O AUC por sujeito so
existe quando ha as duas classes na sessao do sujeito. Saida: subject_report_*.
"""

preds = pd.read_csv(PRED_PATH)

for m in sorted(preds['model'].unique()):
    rows = []
    for subj, d in preds[preds.model == m].groupby('subject'):
        both = d['y_true'].nunique() > 1
        rows.append({
            'subject': subj,
            'n': len(d),
            'n_stress': int((d.y_true == 1).sum()),
            'acc': (d.y_true == d.y_pred).mean(),
            'recall_stress': recall_score(d.y_true, d.y_pred, zero_division=0),
            'auc': roc_auc_score(d.y_true, d.y_prob) if both else np.nan,
            'mean_prob': d.y_prob.mean(),
        })
    rep = pd.DataFrame(rows).set_index('subject').sort_values('auc')
    print(f"\n{m}: sujeitos ordenados por AUC intra-sujeito")
    print(rep.round(3).to_string())
    rep.to_csv(RESULTS_DIR / f"subject_report_{m}.csv")