#%%
import os
import io
import pickle
import psutil
import time
import itertools

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, roc_auc_score, f1_score, precision_score, recall_score, auc

# %%
#%%[Markdown]
# Caminhos
BASE_DIR = r"C:\DEV\mestrado\WSD\experiments"

CSV_PATH = os.path.join(BASE_DIR, "processed", "dataset_stress.csv")
BASE_OUTPUT_DIR = BASE_DIR
FOLDS_DIR = os.path.join(BASE_OUTPUT_DIR, "folds")
INTERNAL_FOLDS_DIR = os.path.join(FOLDS_DIR, "internal_folds")
GRID_SEARCH_DIR = os.path.join(FOLDS_DIR, "grid_search")

os.makedirs(GRID_SEARCH_DIR, exist_ok=True)

#%%
print(f"INTERNAL_FOLDS_DIR: {INTERNAL_FOLDS_DIR}")

# %%
TARGET = "label"
metric = "f1"

param_grid = {
    'n_estimators': [150, 200, 250],
    'max_features': [0.6, 0.7, 0.8],
    'min_samples_leaf': [1, 2, 3]
}

param_names = list(param_grid.keys())
param_values = list(param_grid.values())
all_combinations = list(itertools.product(*param_values))
df_combinations = pd.DataFrame(all_combinations, columns=param_names)
df_combinations['n_estimators'] = df_combinations['n_estimators'].astype(int)
df_combinations['min_samples_leaf'] = df_combinations['min_samples_leaf'].astype(int)

folds = []
for f in os.listdir(INTERNAL_FOLDS_DIR):
    if f.startswith("STRESS_train_") and f.endswith(".csv"):
        fold_number = f.replace("STRESS_train_", "").replace(".csv", "")
        folds.append(int(fold_number))
folds = sorted(folds)
print("Folds: ", folds)


rows_folds = []
rows_summary = []

# %%
# FOR EXTERNO PARA A COMBINAÇÃO
for comb_id, combo in df_combinations.iterrows():

    print(f"Avaliando combinação {comb_id}")
    print(combo.to_dict())

    metrics_folds = []

    # FOR INTERNO PARA A VALIDAÇÃO
    for fold in folds:

        train_path = os.path.join(INTERNAL_FOLDS_DIR, f"STRESS_train_{fold}.csv")
        val_path = os.path.join(INTERNAL_FOLDS_DIR, f"STRESS_val_{fold}.csv")

        df_train = pd.read_csv(train_path)
        df_val = pd.read_csv(val_path)

        # Colunas a remover (ajuste conforme seu dataset)
        cols_to_drop = [TARGET, 'subject_id', 'window_id', 'scenario', 'protocol']  
        cols_to_drop = [c for c in cols_to_drop if c in df_train.columns]

        x_train = df_train.drop(columns=cols_to_drop)
        y_train = df_train[TARGET]

        x_val = df_val.drop(columns=cols_to_drop)
        y_val = df_val[TARGET]

        model = RandomForestClassifier(
            n_estimators=int(combo['n_estimators']),
            max_features=float(combo['max_features']),
            min_samples_leaf=int(combo['min_samples_leaf']),
            random_state=42,
            n_jobs=-1
        )

        # TREINA O MODELO
        # SALVA INFO DE TEMPO DE TREINAMENTO
        start = time.time()
        model.fit(x_train, y_train)
        end = time.time()
        runtime_train = end - start

        # AVALIAÇÃO - PREDICT
        # SALVA TEMPO DE PREDICT
        start = time.time()
        predicoes_binarias = model.predict(x_val)
        end = time.time()
        runtime_bin = end - start

        # AVALIAÇÃO - PREDICT PROBA
        # SALVA TEMPO DE PREDICT
        start = time.time()
        predicoes_proba = model.predict_proba(x_val)[:, 1]  # Apenas classe positiva
        end = time.time()
        runtime_proba = end - start

        # tamanho do modelo em memória
        model_buffer = io.BytesIO()
        pickle.dump(model, model_buffer)
        model_size = model_buffer.tell()

        # Calcular AUPRC corretamente
        roc_auc = roc_auc_score(y_val, predicoes_proba)
        f1 = f1_score(y_val, predicoes_binarias)
        recall = recall_score(y_val, predicoes_binarias)
        precision = precision_score(y_val, predicoes_binarias)
        precision_curve, recall_curve, _ = precision_recall_curve(y_val, predicoes_proba)
        auc_prc = auc(recall_curve, precision_curve)

        metrics_folds.append({
            "combination_id": comb_id,
            "fold": fold,
            "roc_auc": roc_auc,
            "auprc": auc_prc, 
            "f1": f1,
            "precision_curve": precision_curve,
            "recall_curve": recall_curve,
            "precision": precision,
            "recall": recall,
            "runtime_train": runtime_train,
            "runtime_bin": runtime_bin,
            "runtime_proba": runtime_proba,
            "model_size": model_size
        })

    rows_folds.extend(metrics_folds)

    rows_summary.append({
        "combo_id": comb_id,
        **combo.to_dict(),
        "roc_auc": np.mean([m["roc_auc"] for m in metrics_folds]),
        "auprc": np.mean([m["auprc"] for m in metrics_folds]),
        "f1": np.mean([m["f1"] for m in metrics_folds]),
        #"precision_curve": np.mean([m["precision_curve"] for m in metrics_folds]),
        #"recall_curve": np.mean([m["recall_curve"] for m in metrics_folds]),
        "precision": np.mean([m["precision"] for m in metrics_folds]),
        "recall": np.mean([m["recall"] for m in metrics_folds]),
        "runtime_train": np.mean([m["runtime_train"] for m in metrics_folds]),
        "runtime_bin": np.mean([m["runtime_bin"] for m in metrics_folds]),
        "runtime_proba": np.mean([m["runtime_proba"] for m in metrics_folds])
    })


# %%
df_summary = pd.DataFrame(rows_summary)
df_folds = pd.DataFrame(rows_folds)

display(df_summary)

df_summary.to_csv(os.path.join(GRID_SEARCH_DIR, "grid_search_summary.csv"), index=False)
df_folds.to_csv(os.path.join(GRID_SEARCH_DIR, "grid_search_folds.csv"), index=False)


best = df_summary.loc[df_summary[metric].idxmax()]
display(best)

# %% Modelo Regressão Final

