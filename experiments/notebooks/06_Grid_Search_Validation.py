#%%
import os
import pandas as pd
import joblib
import shutil
import time
import io
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc
)
from sklearn.model_selection import learning_curve, PredefinedSplit

#%% [Markdown]

#%%
# Caminhos
BASE_DIR = r"C:\DEV\mestrado\WSD\experiments"
FOLDS_DIR = os.path.join(BASE_DIR, "folds", "external_folds")
GRID_SEARCH_FILE = os.path.join(BASE_DIR, "folds", "grid_search", "grid_search_summary.csv")
OUTPUT_MODELS_DIR = os.path.join(BASE_DIR, "models", "final_validation")
PLOTS_DIR = os.path.join(OUTPUT_MODELS_DIR, "plots")

os.makedirs(OUTPUT_MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

#%%
# Carrega e converte os melhores parâmetros do CSV de Grid Search
def get_best_params(csv_path):

    df = pd.read_csv(csv_path)
    best = df.sort_values(by="f1", ascending=False).iloc[0]
    
    return {
        'n_estimators': int(best['n_estimators']),
        'max_features': float(best['max_features']),
        'min_samples_leaf': int(best['min_samples_leaf'])
    }

def plot_learning_curve_fold(estimator, X_train, y_train, X_test, y_test, fold_idx, save_dir):
    """Gera e salva a curva de aprendizado para um fold específico usando o Test set como validação."""
    
    # Combina treino e teste para usar PredefinedSplit (garante que validação = teste do fold)
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    y_combined = pd.concat([y_train, y_test], ignore_index=True)
    
    # -1 indica amostra de treino, 0 indica amostra de teste
    test_fold = [-1] * len(X_train) + [0] * len(X_test)
    ps = PredefinedSplit(test_fold)
    
    # Calcula a curva
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X_combined, y_combined, cv=ps, 
        n_jobs=-1, train_sizes=np.linspace(0.1, 1.0, 5), scoring='f1'
    )
    
    # Estatísticas
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training Score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Validation Score (Test Fold)")
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color="g")
    plt.title(f"Learning Curve - Fold {fold_idx} (F1-Score)")
    plt.xlabel("Training Examples")
    plt.ylabel("F1 Score")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f"learning_curve_fold{fold_idx}.png"))
    plt.close()

def train_evaluate(fold_idx, params):
    """Treina o modelo em um fold específico e retorna métricas e o artefato."""

    train_df = pd.read_csv(os.path.join(FOLDS_DIR, f"STRESS_fold{fold_idx}_train.csv"))
    test_df = pd.read_csv(os.path.join(FOLDS_DIR, f"STRESS_fold{fold_idx}_test.csv"))
    

    cols_meta = [
                'subject_id', 
                'window_id', 
                'label', 
                'scenario', 
                'protocol'
    ]
    X_train = train_df.drop(columns=[c for c in cols_meta if c in train_df.columns])
    y_train = train_df['label']
    X_test = test_df.drop(columns=[c for c in cols_meta if c in test_df.columns])
    y_test = test_df['label']
    
    clf = RandomForestClassifier(**params, random_state=42, n_jobs=-1)

    # Gera gráfico de aprendizado antes do fit final
    plot_learning_curve_fold(clf, X_train, y_train, X_test, y_test, fold_idx, PLOTS_DIR)

    start_train = time.time()
    clf.fit(X_train, y_train)
    runtime_train = time.time() - start_train
    start_bin = time.time()
    y_pred = clf.predict(X_test)
    runtime_bin = time.time() - start_bin
    start_proba = time.time()
    y_proba = clf.predict_proba(X_test)[:, 1]
    runtime_proba = time.time() - start_proba
    

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_proba)
    auprc = auc(recall_vals, precision_vals)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    

    return clf, {
        'fold': fold_idx,
        'accuracy': acc,
        'f1': f1,
        'roc_auc': roc_auc,
        'auprc': auprc,
        'precision': precision,
        'recall': recall,
        'runtime_train': runtime_train,
        'runtime_bin': runtime_bin,
        'runtime_proba': runtime_proba,
    }

#%%
#
best_params = get_best_params(GRID_SEARCH_FILE)
print(f"{best_params}")

resultados = []
folds_existentes = [f for f in os.listdir(FOLDS_DIR) if "_train.csv" in f]

for i in range(1, len(folds_existentes) + 1):
    print(f"Fold {i}...", end=" ")
    model, metrics = train_evaluate(i, best_params)
    model_path = os.path.join(OUTPUT_MODELS_DIR, f"rf_final_fold{i}.joblib")
    joblib.dump(model, model_path)
    model_size_mb = os.path.getsize(model_path) / (1024 * 1024)
    metrics['model_size_mb'] = model_size_mb
    resultados.append(metrics)
    
    print(f"f1: {metrics['f1']:.4f})")

#%%

df_results = pd.DataFrame(resultados)


print(df_results.round(4).to_string(index=False))
display(df_results.mean(numeric_only=True))

df_results.to_csv(os.path.join(OUTPUT_MODELS_DIR, "metrics_summary.csv"), index=False)

best_row = df_results.loc[df_results['f1'].idxmax()]
best_fold = int(best_row['fold'])
print(f"\nMelhor fold: {best_fold} (F1: {best_row['f1']:.4f})")

src_model = os.path.join(OUTPUT_MODELS_DIR, f"rf_final_fold{best_fold}.joblib")
dst_model = os.path.join(OUTPUT_MODELS_DIR, "best_model.joblib")
shutil.copy(src_model, dst_model)

# %%
