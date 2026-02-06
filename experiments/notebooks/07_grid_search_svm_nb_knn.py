# %%
import os
import io
import time
import pickle
import itertools
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, 
    f1_score, 
    precision_score, 
    recall_score, 
    auc, 
    precision_recall_curve, 
    log_loss, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    RocCurveDisplay
)

BASE_DIR = r"C:\DEV\mestrado\WSD\experiments"
FOLDS_DIR = os.path.join(BASE_DIR, "folds")

EXTERNAL_FOLD_DIR = os.path.join(FOLDS_DIR, "external_folds")
INTERNAL_FOLDS_DIR = os.path.join(FOLDS_DIR, "internal_folds")

# Diretórios específicos para cada modelo
GRID_SEARCH_SVM_DIR = os.path.join(FOLDS_DIR, "grid_search_svm")
GRID_SEARCH_KNN_DIR = os.path.join(FOLDS_DIR, "grid_search_knn")
GRID_SEARCH_NB_DIR = os.path.join(FOLDS_DIR, "grid_search_nb")

os.makedirs(GRID_SEARCH_SVM_DIR, exist_ok=True)
os.makedirs(GRID_SEARCH_KNN_DIR, exist_ok=True)
os.makedirs(GRID_SEARCH_NB_DIR, exist_ok=True)

TARGET = "label"
base_seed = 42

# %% Data Loader
def fold_loader(path_csv):
    df = pd.read_csv(path_csv)
    cols_to_drop = ['subject_id', 'window_id', 'label', 'scenario', 'protocol']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET]
    return X, y

# %%[Markdown]
# ==================== SVM ====================

# %%[Markdown]
# PARAMETROS SVM

param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'kernel': ['rbf', 'linear', 'poly'],
    'gamma': ['scale', 'auto', 0.001, 0.01]
}

# %% GRID SEARCH SVM

def grid_search_svm(INTERNAL_FOLDS_DIR, param_grid, metric_sort="f1"):
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    df_combinations = pd.DataFrame(all_combinations, columns=param_names)
    
    folds = sorted([
        int(f.replace("STRESS_train_", "").replace(".csv", ""))
        for f in os.listdir(INTERNAL_FOLDS_DIR) 
        if f.startswith("STRESS_train_") and f.endswith(".csv")
    ])
    
    print(f"Folds Internos Identificados (SVM): {folds}")
    
    rows_summary = []
    rows_folds = []

    for comb_id, combo in tqdm(df_combinations.iterrows(), total=len(df_combinations), desc="Avaliando Combinações SVM"):
        params = combo.to_dict()
        if 'C' in params: params['C'] = float(params['C'])
        if 'gamma' in params and params['gamma'] not in ['scale', 'auto']:
            params['gamma'] = float(params['gamma'])
        
        metrics_folds = []

        for fold in folds:
            train_path = os.path.join(INTERNAL_FOLDS_DIR, f"STRESS_train_{fold}.csv")
            val_path = os.path.join(INTERNAL_FOLDS_DIR, f"STRESS_val_{fold}.csv")
            
            X_train, y_train = fold_loader(train_path)
            X_val, y_val = fold_loader(val_path)

            # Normalização essencial para SVM
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = SVC(random_state=42, probability=True, **params)
            
            start = time.time()
            model.fit(X_train_scaled, y_train)
            runtime_train = time.time() - start
            
            start = time.time()
            y_pred = model.predict(X_val_scaled)
            y_proba = model.predict_proba(X_val_scaled)[:, 1]
            runtime_inf = time.time() - start
            
            buffer = io.BytesIO()
            pickle.dump({'model': model, 'scaler': scaler}, buffer)
            model_size = buffer.tell()

            precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_proba)
            roc_auc = roc_auc_score(y_val, y_proba)
            auprc = auc(recall_curve, precision_curve)
            
            metrics = {
                "combination_id": comb_id,
                "fold": fold,
                "roc_auc": roc_auc,
                "auprc": auprc,
                "f1": f1_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred),
                "recall": recall_score(y_val, y_pred),
                "runtime_train": runtime_train,
                "runtime_inf": runtime_inf,
                "model_size": model_size
            }
            metrics_folds.append(metrics)

        avg_metrics = pd.DataFrame(metrics_folds).mean(numeric_only=True).to_dict()
        summary_row = {**{"combo_id": comb_id}, **params, **avg_metrics}
        rows_summary.append(summary_row)
        rows_folds.extend(metrics_folds)

    df_summary = pd.DataFrame(rows_summary)
    df_folds = pd.DataFrame(rows_folds)
    
    df_summary.to_csv(os.path.join(GRID_SEARCH_SVM_DIR, "grid_search_summary.csv"), index=False)
    df_folds.to_csv(os.path.join(GRID_SEARCH_SVM_DIR, "grid_search_folds_detail.csv"), index=False)
    
    df_summary = df_summary.sort_values(by=metric_sort, ascending=False)
    
    display(df_summary.head())
    
    best_combo = df_summary.iloc[0]
    print("Melhores parâmetros SVM:", best_combo[param_names].to_dict())
    
    return best_combo[param_names].to_dict()

best_params_svm = grid_search_svm(INTERNAL_FOLDS_DIR, param_grid_svm, metric_sort="f1")

# %% Pareto Front SVM

df_svm = pd.read_csv(os.path.join(GRID_SEARCH_SVM_DIR, "grid_search_summary.csv"))

def pareto_front(df, maximize='f1', minimize=['runtime_train', 'model_size']):
    df = df.copy()
    df['_dominated'] = False
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j: 
                continue
            better_f1 = df.loc[j, maximize] >= df.loc[i, maximize]
            better_costs = all(df.loc[j, c] <= df.loc[i, c] for c in minimize)
            strictly_better = (df.loc[j, maximize] > df.loc[i, maximize]) or \
                             any(df.loc[j, c] < df.loc[i, c] for c in minimize)
            if better_f1 and better_costs and strictly_better:
                df.loc[i, '_dominated'] = True
                break
    return df[~df['_dominated']].drop(columns=['_dominated'])

pareto_svm = pareto_front(df_svm, maximize='f1', minimize=['runtime_train', 'model_size'])

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(df_svm['runtime_train'], df_svm['f1'], 
           c='gray', alpha=0.5, s=40, label='Todas combinações')
plt.scatter(pareto_svm['runtime_train'], pareto_svm['f1'], 
           c='red', s=100, edgecolors='black', linewidth=1.5, 
           label='Fronteira de Pareto', zorder=5)
plt.xlabel('Tempo de Treino (s)', fontsize=11)
plt.ylabel('F1 Score', fontsize=11)
plt.title('SVM: F1 vs Tempo de Treino', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(df_svm['model_size']/1e6, df_svm['f1'], 
           c='gray', alpha=0.5, s=40, label='Todas combinações')
plt.scatter(pareto_svm['model_size']/1e6, pareto_svm['f1'], 
           c='red', s=100, edgecolors='black', linewidth=1.5, 
           label='Fronteira de Pareto', zorder=5)
plt.xlabel('Tamanho do Modelo (MB)', fontsize=11)
plt.ylabel('F1 Score', fontsize=11)
plt.title('SVM: F1 vs Tamanho do Modelo', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% Avaliação Externa SVM

def evaluate_model_svm(EXTERNAL_FOLD_DIR, best_params):
    folds = sorted([
        int(f.replace("STRESS_fold", "").replace("_train.csv", ""))
        for f in os.listdir(EXTERNAL_FOLD_DIR) 
        if f.startswith("STRESS_fold") and f.endswith("_train.csv")
    ])
    
    print(f"Folds Externos Identificados (SVM): {folds}")
    
    rows_eval = []
    
    if 'C' in best_params: best_params['C'] = float(best_params['C'])
    if 'gamma' in best_params and best_params['gamma'] not in ['scale', 'auto']:
        best_params['gamma'] = float(best_params['gamma'])

    for fold in tqdm(folds, desc="Folds Externos SVM"):
        train_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_train.csv")
        test_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_test.csv")
        
        X_train, y_train = fold_loader(train_path)
        X_test, y_test = fold_loader(test_path)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = SVC(random_state=42, probability=True, **best_params)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        
        rows_eval.append({
            "fold": fold,
            "roc_auc": roc_auc_score(y_test, y_proba),
            "auprc": auc(recall_curve, precision_curve),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        })
    
    df_results = pd.DataFrame(rows_eval)
    df_results.to_csv(os.path.join(GRID_SEARCH_SVM_DIR, "evaluation_results.csv"), index=False)
    
    print("\nMédias SVM:")
    display(df_results.mean(numeric_only=True))
    return df_results

results_svm = evaluate_model_svm(EXTERNAL_FOLD_DIR, best_params_svm)

# %%[Markdown]
# ==================== KNN ====================

# %%[Markdown]
# PARAMETROS KNN

param_grid_knn = {
    'n_neighbors': [3, 5, 7, 9, 11, 15],
    'weights': ['uniform', 'distance'],
    'metric': ['euclidean', 'manhattan', 'minkowski']
}

# %% GRID SEARCH KNN

def grid_search_knn(INTERNAL_FOLDS_DIR, param_grid, metric_sort="f1"):
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    df_combinations = pd.DataFrame(all_combinations, columns=param_names)
    
    folds = sorted([
        int(f.replace("STRESS_train_", "").replace(".csv", ""))
        for f in os.listdir(INTERNAL_FOLDS_DIR) 
        if f.startswith("STRESS_train_") and f.endswith(".csv")
    ])
    
    print(f"Folds Internos Identificados (KNN): {folds}")
    
    rows_summary = []
    rows_folds = []

    for comb_id, combo in tqdm(df_combinations.iterrows(), total=len(df_combinations), desc="Avaliando Combinações KNN"):
        params = combo.to_dict()
        if 'n_neighbors' in params: params['n_neighbors'] = int(params['n_neighbors'])
        
        metrics_folds = []

        for fold in folds:
            train_path = os.path.join(INTERNAL_FOLDS_DIR, f"STRESS_train_{fold}.csv")
            val_path = os.path.join(INTERNAL_FOLDS_DIR, f"STRESS_val_{fold}.csv")
            
            X_train, y_train = fold_loader(train_path)
            X_val, y_val = fold_loader(val_path)

            # Normalização essencial para KNN
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = KNeighborsClassifier(n_jobs=-1, **params)
            
            start = time.time()
            model.fit(X_train_scaled, y_train)
            runtime_train = time.time() - start
            
            start = time.time()
            y_pred = model.predict(X_val_scaled)
            y_proba = model.predict_proba(X_val_scaled)[:, 1]
            runtime_inf = time.time() - start
            
            buffer = io.BytesIO()
            pickle.dump({'model': model, 'scaler': scaler}, buffer)
            model_size = buffer.tell()

            precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_proba)
            roc_auc = roc_auc_score(y_val, y_proba)
            auprc = auc(recall_curve, precision_curve)
            
            metrics = {
                "combination_id": comb_id,
                "fold": fold,
                "roc_auc": roc_auc,
                "auprc": auprc,
                "f1": f1_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred),
                "recall": recall_score(y_val, y_pred),
                "runtime_train": runtime_train,
                "runtime_inf": runtime_inf,
                "model_size": model_size
            }
            metrics_folds.append(metrics)

        avg_metrics = pd.DataFrame(metrics_folds).mean(numeric_only=True).to_dict()
        summary_row = {**{"combo_id": comb_id}, **params, **avg_metrics}
        rows_summary.append(summary_row)
        rows_folds.extend(metrics_folds)

    df_summary = pd.DataFrame(rows_summary)
    df_folds = pd.DataFrame(rows_folds)
    
    df_summary.to_csv(os.path.join(GRID_SEARCH_KNN_DIR, "grid_search_summary.csv"), index=False)
    df_folds.to_csv(os.path.join(GRID_SEARCH_KNN_DIR, "grid_search_folds_detail.csv"), index=False)
    
    df_summary = df_summary.sort_values(by=metric_sort, ascending=False)
    
    display(df_summary.head())
    
    best_combo = df_summary.iloc[0]
    print("Melhores parâmetros KNN:", best_combo[param_names].to_dict())
    
    return best_combo[param_names].to_dict()

best_params_knn = grid_search_knn(INTERNAL_FOLDS_DIR, param_grid_knn, metric_sort="f1")

# %% Pareto Front KNN

df_knn = pd.read_csv(os.path.join(GRID_SEARCH_KNN_DIR, "grid_search_summary.csv"))

pareto_knn = pareto_front(df_knn, maximize='f1', minimize=['runtime_train', 'model_size'])

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(df_knn['runtime_train'], df_knn['f1'], 
           c='gray', alpha=0.5, s=40, label='Todas combinações')
plt.scatter(pareto_knn['runtime_train'], pareto_knn['f1'], 
           c='red', s=100, edgecolors='black', linewidth=1.5, 
           label='Fronteira de Pareto', zorder=5)
plt.xlabel('Tempo de Treino (s)', fontsize=11)
plt.ylabel('F1 Score', fontsize=11)
plt.title('KNN: F1 vs Tempo de Treino', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(df_knn['model_size']/1e6, df_knn['f1'], 
           c='gray', alpha=0.5, s=40, label='Todas combinações')
plt.scatter(pareto_knn['model_size']/1e6, pareto_knn['f1'], 
           c='red', s=100, edgecolors='black', linewidth=1.5, 
           label='Fronteira de Pareto', zorder=5)
plt.xlabel('Tamanho do Modelo (MB)', fontsize=11)
plt.ylabel('F1 Score', fontsize=11)
plt.title('KNN: F1 vs Tamanho do Modelo', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% Avaliação Externa KNN

def evaluate_model_knn(EXTERNAL_FOLD_DIR, best_params):
    folds = sorted([
        int(f.replace("STRESS_fold", "").replace("_train.csv", ""))
        for f in os.listdir(EXTERNAL_FOLD_DIR) 
        if f.startswith("STRESS_fold") and f.endswith("_train.csv")
    ])
    
    print(f"Folds Externos Identificados (KNN): {folds}")
    
    rows_eval = []
    
    if 'n_neighbors' in best_params: best_params['n_neighbors'] = int(best_params['n_neighbors'])

    for fold in tqdm(folds, desc="Folds Externos KNN"):
        train_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_train.csv")
        test_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_test.csv")
        
        X_train, y_train = fold_loader(train_path)
        X_test, y_test = fold_loader(test_path)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = KNeighborsClassifier(n_jobs=-1, **best_params)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        
        rows_eval.append({
            "fold": fold,
            "roc_auc": roc_auc_score(y_test, y_proba),
            "auprc": auc(recall_curve, precision_curve),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        })
    
    df_results = pd.DataFrame(rows_eval)
    df_results.to_csv(os.path.join(GRID_SEARCH_KNN_DIR, "evaluation_results.csv"), index=False)
    
    print("\nMédias KNN:")
    display(df_results.mean(numeric_only=True))
    return df_results

results_knn = evaluate_model_knn(EXTERNAL_FOLD_DIR, best_params_knn)

# %%[Markdown]
# ==================== NAIVE BAYES ====================

# %%[Markdown]
# PARAMETROS NAIVE BAYES (GaussianNB tem poucos hiperparâmetros)

param_grid_nb = {
    'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]
}

# %% GRID SEARCH NAIVE BAYES

def grid_search_nb(INTERNAL_FOLDS_DIR, param_grid, metric_sort="f1"):
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    df_combinations = pd.DataFrame(all_combinations, columns=param_names)
    
    folds = sorted([
        int(f.replace("STRESS_train_", "").replace(".csv", ""))
        for f in os.listdir(INTERNAL_FOLDS_DIR) 
        if f.startswith("STRESS_train_") and f.endswith(".csv")
    ])
    
    print(f"Folds Internos Identificados (NB): {folds}")
    
    rows_summary = []
    rows_folds = []

    for comb_id, combo in tqdm(df_combinations.iterrows(), total=len(df_combinations), desc="Avaliando Combinações NB"):
        params = combo.to_dict()
        if 'var_smoothing' in params: params['var_smoothing'] = float(params['var_smoothing'])
        
        metrics_folds = []

        for fold in folds:
            train_path = os.path.join(INTERNAL_FOLDS_DIR, f"STRESS_train_{fold}.csv")
            val_path = os.path.join(INTERNAL_FOLDS_DIR, f"STRESS_val_{fold}.csv")
            
            X_train, y_train = fold_loader(train_path)
            X_val, y_val = fold_loader(val_path)

            # Naive Bayes assume features Gaussianas - normalização ajuda
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)

            model = GaussianNB(**params)
            
            start = time.time()
            model.fit(X_train_scaled, y_train)
            runtime_train = time.time() - start
            
            start = time.time()
            y_pred = model.predict(X_val_scaled)
            y_proba = model.predict_proba(X_val_scaled)[:, 1]
            runtime_inf = time.time() - start
            
            buffer = io.BytesIO()
            pickle.dump({'model': model, 'scaler': scaler}, buffer)
            model_size = buffer.tell()

            precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_proba)
            roc_auc = roc_auc_score(y_val, y_proba)
            auprc = auc(recall_curve, precision_curve)
            
            metrics = {
                "combination_id": comb_id,
                "fold": fold,
                "roc_auc": roc_auc,
                "auprc": auprc,
                "f1": f1_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred),
                "recall": recall_score(y_val, y_pred),
                "runtime_train": runtime_train,
                "runtime_inf": runtime_inf,
                "model_size": model_size
            }
            metrics_folds.append(metrics)

        avg_metrics = pd.DataFrame(metrics_folds).mean(numeric_only=True).to_dict()
        summary_row = {**{"combo_id": comb_id}, **params, **avg_metrics}
        rows_summary.append(summary_row)
        rows_folds.extend(metrics_folds)

    df_summary = pd.DataFrame(rows_summary)
    df_folds = pd.DataFrame(rows_folds)
    
    df_summary.to_csv(os.path.join(GRID_SEARCH_NB_DIR, "grid_search_summary.csv"), index=False)
    df_folds.to_csv(os.path.join(GRID_SEARCH_NB_DIR, "grid_search_folds_detail.csv"), index=False)
    
    df_summary = df_summary.sort_values(by=metric_sort, ascending=False)
    
    display(df_summary)
    
    best_combo = df_summary.iloc[0]
    print("Melhores parâmetros Naive Bayes:", best_combo[param_names].to_dict())
    
    return best_combo[param_names].to_dict()

best_params_nb = grid_search_nb(INTERNAL_FOLDS_DIR, param_grid_nb, metric_sort="f1")

# %% Pareto Front Naive Bayes

df_nb = pd.read_csv(os.path.join(GRID_SEARCH_NB_DIR, "grid_search_summary.csv"))

pareto_nb = pareto_front(df_nb, maximize='f1', minimize=['runtime_train', 'model_size'])

plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.scatter(df_nb['runtime_train'], df_nb['f1'], 
           c='gray', alpha=0.5, s=40, label='Todas combinações')
plt.scatter(pareto_nb['runtime_train'], pareto_nb['f1'], 
           c='red', s=100, edgecolors='black', linewidth=1.5, 
           label='Fronteira de Pareto', zorder=5)
plt.xlabel('Tempo de Treino (s)', fontsize=11)
plt.ylabel('F1 Score', fontsize=11)
plt.title('Naive Bayes: F1 vs Tempo de Treino', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.scatter(df_nb['model_size']/1e6, df_nb['f1'], 
           c='gray', alpha=0.5, s=40, label='Todas combinações')
plt.scatter(pareto_nb['model_size']/1e6, pareto_nb['f1'], 
           c='red', s=100, edgecolors='black', linewidth=1.5, 
           label='Fronteira de Pareto', zorder=5)
plt.xlabel('Tamanho do Modelo (MB)', fontsize=11)
plt.ylabel('F1 Score', fontsize=11)
plt.title('Naive Bayes: F1 vs Tamanho do Modelo', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# %% Avaliação Externa Naive Bayes

def evaluate_model_nb(EXTERNAL_FOLD_DIR, best_params):
    folds = sorted([
        int(f.replace("STRESS_fold", "").replace("_train.csv", ""))
        for f in os.listdir(EXTERNAL_FOLD_DIR) 
        if f.startswith("STRESS_fold") and f.endswith("_train.csv")
    ])
    
    print(f"Folds Externos Identificados (NB): {folds}")
    
    rows_eval = []
    
    if 'var_smoothing' in best_params: best_params['var_smoothing'] = float(best_params['var_smoothing'])

    for fold in tqdm(folds, desc="Folds Externos NB"):
        train_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_train.csv")
        test_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_test.csv")
        
        X_train, y_train = fold_loader(train_path)
        X_test, y_test = fold_loader(test_path)
        
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        model = GaussianNB(**best_params)
        model.fit(X_train_scaled, y_train)
        
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
        
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
        
        rows_eval.append({
            "fold": fold,
            "roc_auc": roc_auc_score(y_test, y_proba),
            "auprc": auc(recall_curve, precision_curve),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        })
    
    df_results = pd.DataFrame(rows_eval)
    df_results.to_csv(os.path.join(GRID_SEARCH_NB_DIR, "evaluation_results.csv"), index=False)
    
    print("\nMédias Naive Bayes:")
    display(df_results.mean(numeric_only=True))
    return df_results

results_nb = evaluate_model_nb(EXTERNAL_FOLD_DIR, best_params_nb)

# %%[Markdown]
# ==================== COMPARAÇÃO FINAL ====================

# %% Comparação de todos os modelos

def compare_all_models():
    # Carregar resultados
    rf_results = pd.read_csv(os.path.join(FOLDS_DIR, "grid_search", "Evaluation_results.csv"))
    svm_results = pd.read_csv(os.path.join(GRID_SEARCH_SVM_DIR, "evaluation_results.csv"))
    knn_results = pd.read_csv(os.path.join(GRID_SEARCH_KNN_DIR, "evaluation_results.csv"))
    nb_results = pd.read_csv(os.path.join(GRID_SEARCH_NB_DIR, "evaluation_results.csv"))
    
    comparison = pd.DataFrame({
        'Random Forest': rf_results.mean(numeric_only=True),
        'SVM': svm_results.mean(numeric_only=True),
        'KNN': knn_results.mean(numeric_only=True),
        'Naive Bayes': nb_results.mean(numeric_only=True)
    }).round(4)
    
    print("=== COMPARAÇÃO DE MODELOS (Médias dos Folds Externos) ===")
    display(comparison)
    
    # Plot comparativo
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    metrics = ['f1', 'roc_auc', 'precision', 'recall']
    titles = ['F1 Score', 'ROC AUC', 'Precision', 'Recall']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        models = ['RF', 'SVM', 'KNN', 'NB']
        values = [
            rf_results[metric].mean(),
            svm_results[metric].mean(),
            knn_results[metric].mean(),
            nb_results[metric].mean()
        ]
        stds = [
            rf_results[metric].std(),
            svm_results[metric].std(),
            knn_results[metric].std(),
            nb_results[metric].std()
        ]
        
        bars = ax.bar(models, values, yerr=stds, capsize=5, 
                     color=['purple', 'red', 'blue', 'green'], alpha=0.7)
        ax.set_ylim(0, 1.1)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel('Score')
        ax.grid(axis='y', alpha=0.3)
        
        # Adicionar valores nas barras
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Comparação de Modelos - Métricas Médias', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    return comparison

final_comparison = compare_all_models()

# %% Matrizes de Confusão e ROC para todos os modelos

def plot_all_confusion_matrices():
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    models_data = [
        ("Random Forest", all_y_test_rf, all_y_pred_rf, 'Purples'),
        ("SVM", all_y_test_svm, all_y_pred_svm, 'Reds'),
        ("KNN", all_y_test_knn, all_y_pred_knn, 'Blues'),
        ("Naive Bayes", all_y_test_nb, all_y_pred_nb, 'Greens')
    ]
    
    for idx, (name, y_test, y_pred, cmap) in enumerate(models_data):
        ax = axes[idx // 2, idx % 2]
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Rest (0)', 'Stress (1)'])
        disp.plot(ax=ax, cmap=cmap, values_format='d')
        ax.set_title(f'Matriz de Confusão - {name}', fontsize=12, fontweight='bold')
        ax.grid(False)
    
    plt.tight_layout()
    plt.show()

def plot_all_roc_curves():
    fig, ax = plt.subplots(figsize=(8, 6))
    
    models_roc = [
        ("Random Forest", all_y_test_rf, all_y_proba_rf, "purple"),
        ("SVM", all_y_test_svm, all_y_proba_svm, "red"),
        ("KNN", all_y_test_knn, all_y_proba_knn, "blue"),
        ("Naive Bayes", all_y_test_nb, all_y_proba_nb, "green")
    ]
    
    for name, y_test, y_proba, color in models_roc:
        RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax, name=name, color=color, lw=2)
    
    ax.plot([0, 1], [0, 1], "k--", lw=2, label="Aleatório")
    ax.set_title("Curvas ROC - Comparação de Modelos", fontsize=14, fontweight='bold')
    ax.set_xlabel("Taxa de Falsos Positivos")
    ax.set_ylabel("Taxa de Verdadeiros Positivos")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.show()

# %% Execução das avaliações finais para visualizações

# Recoletar predições para todos os modelos (necessário para os plots)
def collect_predictions(model_type, params, needs_scaler=True):
    folds = sorted([
        int(f.replace("STRESS_fold", "").replace("_train.csv", ""))
        for f in os.listdir(EXTERNAL_FOLD_DIR) 
        if f.startswith("STRESS_fold") and f.endswith("_train.csv")
    ])
    
    all_y_test = []
    all_y_pred = []
    all_y_proba = []
    
    for fold in tqdm(folds, desc=f"Coletando {model_type}"):
        train_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_train.csv")
        test_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_test.csv")
        
        X_train, y_train = fold_loader(train_path)
        X_test, y_test = fold_loader(test_path)
        
        if needs_scaler:
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)
        
        if model_type == "SVM":
            model = SVC(random_state=42, probability=True, **params)
        elif model_type == "KNN":
            model = KNeighborsClassifier(n_jobs=-1, **params)
        elif model_type == "NB":
            model = GaussianNB(**params)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
    
    return all_y_test, all_y_pred, all_y_proba

# Coletar predições (executar após ter os best_params)
print("Coletando predições para visualizações...")
all_y_test_svm, all_y_pred_svm, all_y_proba_svm = collect_predictions("SVM", best_params_svm)
all_y_test_knn, all_y_pred_knn, all_y_proba_knn = collect_predictions("KNN", best_params_knn)
all_y_test_nb, all_y_pred_nb, all_y_proba_nb = collect_predictions("NB", best_params_nb)

# Plotar comparações
plot_all_confusion_matrices()
plot_all_roc_curves()

# %% Salvar melhores modelos

# SVM
MODEL_SAVE_PATH_SVM = os.path.join(BASE_DIR, "best_svm_model.pkl")
X_final, y_final = fold_loader(os.path.join(EXTERNAL_FOLD_DIR, "STRESS_fold5_train.csv"))
scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X_final)
final_model_svm = SVC(random_state=42, probability=True, **best_params_svm)
final_model_svm.fit(X_final_scaled, y_final)
with open(MODEL_SAVE_PATH_SVM, 'wb') as f:
    pickle.dump({'model': final_model_svm, 'scaler': scaler_final}, f)
print(f"Modelo SVM salvo em: {MODEL_SAVE_PATH_SVM}")

# KNN
MODEL_SAVE_PATH_KNN = os.path.join(BASE_DIR, "best_knn_model.pkl")
X_final, y_final = fold_loader(os.path.join(EXTERNAL_FOLD_DIR, "STRESS_fold5_train.csv"))
scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X_final)
final_model_knn = KNeighborsClassifier(n_jobs=-1, **best_params_knn)
final_model_knn.fit(X_final_scaled, y_final)
with open(MODEL_SAVE_PATH_KNN, 'wb') as f:
    pickle.dump({'model': final_model_knn, 'scaler': scaler_final}, f)
print(f"Modelo KNN salvo em: {MODEL_SAVE_PATH_KNN}")

# Naive Bayes
MODEL_SAVE_PATH_NB = os.path.join(BASE_DIR, "best_nb_model.pkl")
X_final, y_final = fold_loader(os.path.join(EXTERNAL_FOLD_DIR, "STRESS_fold5_train.csv"))
scaler_final = StandardScaler()
X_final_scaled = scaler_final.fit_transform(X_final)
final_model_nb = GaussianNB(**best_params_nb)
final_model_nb.fit(X_final_scaled, y_final)
with open(MODEL_SAVE_PATH_NB, 'wb') as f:
    pickle.dump({'model': final_model_nb, 'scaler': scaler_final}, f)
print(f"Modelo Naive Bayes salvo em: {MODEL_SAVE_PATH_NB}")

# %%