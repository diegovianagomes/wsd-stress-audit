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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, 
    auc, precision_recall_curve, log_loss
)


BASE_DIR = r"C:\DEV\mestrado\WSD\experiments"
FOLDS_DIR = os.path.join(BASE_DIR, "folds")

EXTERNAL_FOLD_DIR = os.path.join(FOLDS_DIR, "external_folds")
INTERNAL_FOLDS_DIR = os.path.join(FOLDS_DIR, "internal_folds")
GRID_SEARCH_DIR = os.path.join(FOLDS_DIR, "grid_search")

os.makedirs(GRID_SEARCH_DIR, exist_ok=True)

TARGET = "label"


# %% Data Loader
def fold_loader(path_csv):
    df = pd.read_csv(path_csv)

    cols_to_drop = [
                    'subject_id', 
                    'window_id', 
                    'label', 
                    'scenario', 
                    'protocol'
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET]
    return X, y

#%%[Markdown]
# Parametros

param_grid = {
    'n_estimators': [150, 200, 250],
    'max_features': [0.6, 0.7, 0.8],
    'min_samples_leaf': [1, 2, 3]
}

# %%[Markdown]
# Grid Search com os Folds Internos

def grid_search(INTERNAL_FOLDS_DIR, param_grid, metric_sort="f1"):
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    df_combinations = pd.DataFrame(all_combinations, columns=param_names)
    
    folds = sorted([
        int(f.replace("STRESS_train_", "").replace(".csv", ""))
        for f in os.listdir(INTERNAL_FOLDS_DIR) 
        if f.startswith("STRESS_train_") and f.endswith(".csv")
    ])
    
    print(f"Folds Internos Identificados: {folds}")
    
    rows_summary = []
    rows_folds = []

    for comb_id, combo in tqdm(df_combinations.iterrows(), total=len(df_combinations), desc="Avaliando Combinações"):
        params = combo.to_dict()
        
        # Ajuste dos tipos
        if 'n_estimators' in params: params['n_estimators'] = int(params['n_estimators'])
        if 'min_samples_leaf' in params: params['min_samples_leaf'] = int(params['min_samples_leaf'])
        
        metrics_folds = []

        # Loop de Treino de Validação dos folds internos
        for fold in folds:
            train_path = os.path.join(INTERNAL_FOLDS_DIR, f"STRESS_train_{fold}.csv")
            val_path = os.path.join(INTERNAL_FOLDS_DIR, f"STRESS_val_{fold}.csv")
            
            X_train, y_train = fold_loader(train_path)
            X_val, y_val = fold_loader(val_path)

            model = RandomForestClassifier(n_jobs=-1, random_state=42, **params)
            
            # TREINO
            start = time.time()
            model.fit(X_train, y_train)
            runtime_train = time.time() - start
            
            # INFERÊNCIA
            start = time.time()
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            runtime_inf = time.time() - start
            
            # TAMANHO DO MODELO
            buffer = io.BytesIO()
            pickle.dump(model, buffer)
            model_size = buffer.tell()

            # MÉTRICAS
            precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_proba)
            
            metrics = {
                "combination_id": comb_id,
                "fold": fold,
                "roc_auc": roc_auc_score(y_val, y_proba),
                "auprc": auc(recall_curve, precision_curve),
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
    
    df_summary.to_csv(os.path.join(GRID_SEARCH_DIR, "grid_search_summary.csv"), index=False)
    df_folds.to_csv(os.path.join(GRID_SEARCH_DIR, "grid_search_folds_detail.csv"), index=False)
    
    df_summary = df_summary.sort_values(by=metric_sort, ascending=False)
    

    display(df_summary)
    
    best_combo = df_summary.iloc[0]
 
    print(best_combo[param_names])
    
    return best_combo[param_names].to_dict()

best_params = grid_search(INTERNAL_FOLDS_DIR, param_grid, metric_sort="f1")

#%%
# Carregar resultados do grid search do Random Forest
GRID_SEARCH_DIR = r"C:\DEV\mestrado\WSD\experiments\folds\grid_search"
df_rf = pd.read_csv(os.path.join(GRID_SEARCH_DIR, "grid_search_summary.csv"))

def pareto_front(df, maximize='f1', minimize=['runtime_train', 'model_size']):
    df = df.copy()
    df['_dominated'] = False
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j: 
                continue
            # Verifica se j domina i: melhor F1 E menor custo
            better_f1 = df.loc[j, maximize] >= df.loc[i, maximize]
            better_costs = all(df.loc[j, c] <= df.loc[i, c] for c in minimize)
            strictly_better = (df.loc[j, maximize] > df.loc[i, maximize]) or \
                             any(df.loc[j, c] < df.loc[i, c] for c in minimize)
            if better_f1 and better_costs and strictly_better:
                df.loc[i, '_dominated'] = True
                break
    return df[~df['_dominated']].drop(columns=['_dominated'])

pareto_rf = pareto_front(df_rf, maximize='f1', minimize=['runtime_train', 'model_size'])

plt.figure(figsize=(14, 5))

# F1 vs Tempo de Treino
plt.subplot(1, 2, 1)
plt.scatter(df_rf['runtime_train'], df_rf['f1'], 
           c='gray', alpha=0.5, s=40, label='Todas combinações')
plt.scatter(pareto_rf['runtime_train'], pareto_rf['f1'], 
           c='red', s=100, edgecolors='black', linewidth=1.5, 
           label='Fronteira de Pareto', zorder=5)
plt.xlabel('Tempo de Treino (s)', fontsize=11)
plt.ylabel('F1 Score', fontsize=11)
plt.title('Random Forest: F1 vs Tempo de Treino', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

# F1 vs Tamanho do Modelo
plt.subplot(1, 2, 2)
plt.scatter(df_rf['model_size']/1e6, df_rf['f1'], 
           c='gray', alpha=0.5, s=40, label='Todas combinações')
plt.scatter(pareto_rf['model_size']/1e6, pareto_rf['f1'], 
           c='red', s=100, edgecolors='black', linewidth=1.5, 
           label='Fronteira de Pareto', zorder=5)
plt.xlabel('Tamanho do Modelo (MB)', fontsize=11)
plt.ylabel('F1 Score', fontsize=11)
plt.title('Random Forest: F1 vx Tamanho do Modelo', fontsize=12, fontweight='bold')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()


pareto_table = pareto_rf.sort_values('f1', ascending=False)[[
    'combo_id', 'n_estimators', 'max_features', 'min_samples_leaf',
    'f1', 'runtime_train', 'runtime_inf', 'model_size'
]].copy()
pareto_table['model_size_mb'] = (pareto_table['model_size'] / 1e6).round(2)
pareto_table = pareto_table.drop(columns=['model_size']).reset_index(drop=True)

best_f1_row = pareto_table.iloc[0]
print(f"Melhor F1: combo_id={int(best_f1_row['combo_id'])} → F1={best_f1_row['f1']:.4f} "
      f"(n_est={int(best_f1_row['n_estimators'])}, max_feat={best_f1_row['max_features']}, "
      f"min_leaf={int(best_f1_row['min_samples_leaf'])})")

fastest = pareto_table.loc[pareto_table['runtime_train'].idxmin()]
smallest = pareto_table.loc[pareto_table['model_size_mb'].idxmin()]

print(f"\nmenor tempo: combo_id={int(fastest['combo_id'])} → {fastest['runtime_train']:.2f}s (F1={fastest['f1']:.4f})")
print(f"menor tamanho: combo_id={int(smallest['combo_id'])} → {smallest['model_size_mb']:.2f}MB (F1={smallest['f1']:.4f})")
# %%[Markdown] 
# Avaliação com o fold externo
def evaluate_model(EXTERNAL_FOLD_DIR, best_params):
    
    folds = sorted([
        int(f.replace("STRESS_fold", "").replace("_train.csv", ""))
        for f in os.listdir(EXTERNAL_FOLD_DIR) 
        if f.startswith("STRESS_fold") and f.endswith("_train.csv")
    ])
    
    print(f"Folds Externos Identificados: {folds}")
    
    rows_eval = []
    
    # Ajuste dis tipos
    if 'n_estimators' in best_params: best_params['n_estimators'] = int(best_params['n_estimators'])
    if 'min_samples_leaf' in best_params: best_params['min_samples_leaf'] = int(best_params['min_samples_leaf'])

    for fold in tqdm(folds, desc="Folds Externos"):

        train_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_train.csv")
        test_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_test.csv")
        
        X_train, y_train = fold_loader(train_path)
        X_test, y_test = fold_loader(test_path)
        
        model = RandomForestClassifier(n_jobs=-1, random_state=42, **best_params)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Métricas Finais
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
    output_path = os.path.join(GRID_SEARCH_DIR, "Evaluation_results.csv")
    df_results.to_csv(output_path, index=False)
    
    display(df_results.mean(numeric_only=True))

evaluate_model(EXTERNAL_FOLD_DIR, best_params)

#%%
best_params_rf = {
    'n_estimators': 250,      
    'max_features': 0.6,
    'min_samples_leaf': 1,
    'n_jobs': -1,
    'random_state': 42
}
#%%
def plot_learning_curves(EXTERNAL_FOLD_DIR, params):

    folds = sorted([
        int(f.replace("STRESS_fold", "").replace("_train.csv", ""))
        for f in os.listdir(EXTERNAL_FOLD_DIR) 
        if f.startswith("STRESS_fold") and f.endswith("_train.csv")
    ])
    

    n_folds = len(folds)
    n_cols = 2
    n_rows = math.ceil(n_folds / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    step_size = 5 
    max_trees = params['n_estimators']
    

    rf_params = params.copy()
    del rf_params['n_estimators']

    for i, fold in enumerate(tqdm(folds, desc="Processando Folds")):
        ax = axes[i]
        
        train_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_train.csv")
        test_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_test.csv")
        
        X_train, y_train = fold_loader(train_path)
        X_test, y_test = fold_loader(test_path)
        
        train_loss = []
        test_loss = []
        n_trees_axis = []
        

        model = RandomForestClassifier(
            n_estimators=0, 
            warm_start=True, 
            **rf_params
        )
        
        for n_trees in range(step_size, max_trees + 1, step_size):
            model.set_params(n_estimators=n_trees)
            model.fit(X_train, y_train)
            
            proba_train = model.predict_proba(X_train)
            proba_test = model.predict_proba(X_test)
            
            loss_train = log_loss(y_train, proba_train)
            loss_test = log_loss(y_test, proba_test)
            
            train_loss.append(loss_train)
            test_loss.append(loss_test)
            n_trees_axis.append(n_trees)
        
        ax.plot(n_trees_axis, train_loss, label='Treino', color='blue', linewidth=1.5)
        ax.plot(n_trees_axis, test_loss, label='Teste (Validação)', color='orange', linewidth=2)
        
        ax.set_title(f"Fold {fold} (RF)", fontsize=12, fontweight='bold')
        ax.set_ylabel('Log Loss')
        ax.set_xlabel('Número de Árvores (n_estimators)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        min_loss_val = min(test_loss)
        min_loss_idx = test_loss.index(min_loss_val)
        min_loss_trees = n_trees_axis[min_loss_idx]
        
        ax.axvline(min_loss_trees, color='red', linestyle='--', alpha=0.5)
        ax.text(min_loss_trees, min_loss_val, f' Min: {min_loss_val:.3f}', color='red', fontsize=9)


    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


plot_learning_curves(EXTERNAL_FOLD_DIR, best_params_rf)

#%%
def plot_rf_feature_importance(train_path, best_params, top_n=20, figsize=(10, 8)):

    # Carregar dados
    X_train, y_train = fold_loader(train_path)
    
    # Remove n_jobs from best_params if it exists to avoid conflict
    params = best_params.copy()
    params.pop('n_jobs', None)  # Remove n_jobs if present
    
    # Treinar modelo
    model = RandomForestClassifier(
        n_jobs=-1,
        **params
    )
    model.fit(X_train, y_train)
    
    # Extrair importâncias
    importances = model.feature_importances_
    feature_names = X_train.columns
    
    # Criar DataFrame e ordenar
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Selecionar top N
    top_features = feat_imp.head(top_n)
    
    # Plotar
    plt.figure(figsize=figsize)
    sns.barplot(
        data=top_features,
        x='importance',
        y='feature',
        palette='viridis'
    )
    
    plt.title(f'Features mais importantes para Random Forest', fontsize=14, fontweight='bold')
    plt.xlabel('Importância Média (Mean Decrease Impurity)', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.grid(axis='x', alpha=0.3)
    
    # Adicionar valores nas barras
    for i, v in enumerate(top_features['importance']):
        plt.text(v + 0.001, i, f'{v:.4f}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    # Retornar DataFrame completo para análise posterior
    return feat_imp
# %%
# Selecionar um fold externo para análise (ex: fold 0)
train_path = os.path.join(EXTERNAL_FOLD_DIR, "STRESS_fold1_train.csv")

# Parâmetros ótimos do Random Forest
best_params_rf = {
    'n_estimators': 250,      
    'max_features': 0.6,
    'min_samples_leaf': 1,
    'n_jobs': -1,
    'random_state': 42
}

# Gerar visualização das features mais importantes
feature_importance_df = plot_rf_feature_importance(
    train_path=train_path,
    best_params=best_params_rf,
    top_n=20,
    figsize=(10, 8)
)

# Opcional: salvar o ranking completo em CSV
feature_importance_df.to_csv(
    os.path.join(GRID_SEARCH_DIR, "rf_feature_importance.csv"),
    index=False
)
# %%
