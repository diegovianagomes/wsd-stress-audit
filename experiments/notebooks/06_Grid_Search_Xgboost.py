# %% [Markdown] 
#
import os
import io
import matplotlib.pyplot as plt
import seaborn as sns
import math
import time
import pickle
import itertools
import pandas as pd
import numpy as np
from tqdm import tqdm
from xgboost import XGBClassifier 
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score, recall_score, 
    auc, precision_recall_curve
)

# 
BASE_DIR = r"C:\DEV\mestrado\WSD\experiments"
FOLDS_DIR = os.path.join(BASE_DIR, "folds")

EXTERNAL_FOLD_DIR = os.path.join(FOLDS_DIR, "external_folds")
INTERNAL_FOLDS_DIR = os.path.join(FOLDS_DIR, "internal_folds")
GRID_SEARCH_DIR = os.path.join(FOLDS_DIR, "grid_search_xgb") 

os.makedirs(GRID_SEARCH_DIR, exist_ok=True)

TARGET = "label"

# %% [Markdown] 
# Data Loader
def fold_loader(path_csv):
    df = pd.read_csv(path_csv)

    cols_to_drop = ['subject_id', 'window_id', 'label', 'scenario', 'protocol']
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    
    X = df.drop(columns=cols_to_drop)
    y = df[TARGET]
    return X, y

#%% [Markdown] 
# parâmetros para XGBoost

#
'''param_grid_xgb = {
    'n_estimators': [50, 100, 200, 300],
    'learning_rate': [0.01, 0.05, 0.1, 0.3],    
    'max_depth': [3, 5, 7, 9],             
    'subsample': [0.6, 0.8, 1.0]         
}'''

# parametros focando nos melhores 131-179
param_grid_xgb = {
    'n_estimators': [180, 220, 260, 300, 340],
    'learning_rate': [0.08, 0.09, 0.1, 0.11, 0.12],    
    'max_depth': [8, 9, 10],             
    'subsample': [0.9, 1.0]
}

# %% [Markdown] 
# Grid Search 
def grid_search_xgb(INTERNAL_FOLDS_DIR, param_grid, metric_sort="f1"):
        
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    all_combinations = list(itertools.product(*param_values))
    df_combinations = pd.DataFrame(all_combinations, columns=param_names)
    
    folds = sorted([
        int(f.replace("STRESS_train_", "").replace(".csv", ""))
        for f in os.listdir(INTERNAL_FOLDS_DIR) 
        if f.startswith("STRESS_train_") and f.endswith(".csv")
    ])
    
    rows_summary = []
    rows_folds = []

    for comb_id, combo in tqdm(df_combinations.iterrows(), total=len(df_combinations), desc="Combinações XGB"):
        params = combo.to_dict()
        
        if 'n_estimators' in params: params['n_estimators'] = int(params['n_estimators'])
        if 'max_depth' in params: params['max_depth'] = int(params['max_depth'])
        
        metrics_folds = []

        for fold in folds:
            train_path = os.path.join(INTERNAL_FOLDS_DIR, f"STRESS_train_{fold}.csv")
            val_path = os.path.join(INTERNAL_FOLDS_DIR, f"STRESS_val_{fold}.csv")
            
            X_train, y_train = fold_loader(train_path)
            X_val, y_val = fold_loader(val_path)

            model = XGBClassifier(
                n_jobs=-1, 
                random_state=42, 
                use_label_encoder=False, 
                eval_metric='logloss',
                **params
            )
            
            # TREINO
            start = time.time()
            model.fit(X_train, y_train)
            runtime_train = time.time() - start
            
            # INFERÊNCIA
            start = time.time()
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]
            runtime_inf = time.time() - start
            
            # TAMANHO
            buffer = io.BytesIO()
            pickle.dump(model, buffer)
            model_size = buffer.tell()

            # MÉTRICAS
            try:
                prec, rec, _ = precision_recall_curve(y_val, y_proba)
                auprc_val = auc(rec, prec)
                roc_val = roc_auc_score(y_val, y_proba)
            except:
                auprc_val = 0
                roc_val = 0

            metrics = {
                "combination_id": comb_id,
                "fold": fold,
                "roc_auc": roc_val,
                "auprc": auprc_val,
                "f1": f1_score(y_val, y_pred),
                "precision": precision_score(y_val, y_pred),
                "recall": recall_score(y_val, y_pred),
                "runtime_train": runtime_train,
                "runtime_inf": runtime_inf,
                "model_size": model_size
            }
            metrics_folds.append(metrics)

        avg_metrics = pd.DataFrame(metrics_folds).mean(numeric_only=True).to_dict()
        for k in ['fold', 'combination_id']: 
            if k in avg_metrics: del avg_metrics[k]

        summary_row = {**{"combo_id": comb_id}, **params, **avg_metrics}
        rows_summary.append(summary_row)
        rows_folds.extend(metrics_folds)


    df_summary = pd.DataFrame(rows_summary)
    df_folds = pd.DataFrame(rows_folds)
    
    df_summary.to_csv(os.path.join(GRID_SEARCH_DIR, "xgb_grid_summary.csv"), index=False)
    df_folds.to_csv(os.path.join(GRID_SEARCH_DIR, "xgb_folds_detail.csv"), index=False)
    

    df_summary = df_summary.sort_values(by=metric_sort, ascending=False)
    
    print(f"\n--- Resultado do Grid Search XGBoost (Top 5 por {metric_sort}) ---")
    display(df_summary.head(5))
    
    best_combo = df_summary.iloc[0]
    print(f"\nMelhor Combinação XGBoost:")
    print(best_combo[param_names])
    
    return best_combo[param_names].to_dict()

best_params_xgb = grid_search_xgb(INTERNAL_FOLDS_DIR, param_grid_xgb, metric_sort="f1")

# %% [Markdown] 
# Avaliação final 
def evaluate_xgb(EXTERNAL_FOLD_DIR, best_params):

    
    folds = sorted([
        int(f.replace("STRESS_fold", "").replace("_train.csv", ""))
        for f in os.listdir(EXTERNAL_FOLD_DIR) 
        if f.startswith("STRESS_fold") and f.endswith("_train.csv")
    ])
    
    print(f"Folds Externos Identificados: {folds}")
    
    rows_eval = []
    
    if 'n_estimators' in best_params: best_params['n_estimators'] = int(best_params['n_estimators'])
    if 'max_depth' in best_params: best_params['max_depth'] = int(best_params['max_depth'])

    for fold in tqdm(folds, desc="Avaliando Externos"):
        train_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_train.csv")
        test_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_test.csv")
        
        X_train, y_train = fold_loader(train_path)
        X_test, y_test = fold_loader(test_path)
        
        # TREINO FINAL
        model = XGBClassifier(
            n_jobs=-1, 
            random_state=42, 
            use_label_encoder=False, 
            eval_metric='logloss',
            **best_params
        )
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        prec, rec, _ = precision_recall_curve(y_test, y_proba)
        
        rows_eval.append({
            "fold": fold,
            "roc_auc": roc_auc_score(y_test, y_proba),
            "auprc": auc(rec, prec),
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred)
        })
    
    df_results = pd.DataFrame(rows_eval)
    output_path = os.path.join(GRID_SEARCH_DIR, "xgb_final_evaluation.csv")
    df_results.to_csv(output_path, index=False)
    
    display(df_results.mean(numeric_only=True))


evaluate_xgb(EXTERNAL_FOLD_DIR, best_params_xgb)

#%%
# Carregar resultados da grid refinada
df = pd.read_csv(os.path.join(GRID_SEARCH_DIR, "xgb_grid_summary.csv"))

def pareto_front(df, maximize='f1', minimize=['runtime_train', 'model_size']):
    df = df.copy()
    df['_dominated'] = False
    for i in range(len(df)):
        for j in range(len(df)):
            if i == j: continue
            better_metric = df.loc[j, maximize] >= df.loc[i, maximize]
            better_costs = all(df.loc[j, c] <= df.loc[i, c] for c in minimize)
            strictly_better = (df.loc[j, maximize] > df.loc[i, maximize]) or \
                             any(df.loc[j, c] < df.loc[i, c] for c in minimize)
            if better_metric and better_costs and strictly_better:
                df.loc[i, '_dominated'] = True
                break
    return df[~df['_dominated']].drop(columns=['_dominated'])

pareto = pareto_front(df, maximize='f1', minimize=['runtime_train', 'model_size'])

# Visualização
plt.figure(figsize=(12, 5))

# F1 vs Tempo de Treino
plt.subplot(1, 2, 1)
plt.scatter(df['runtime_train'], df['f1'], c='gray', alpha=0.4, s=30, label='Todas combinações')
plt.scatter(pareto['runtime_train'], pareto['f1'], c='red', s=80, 
            edgecolors='black', linewidth=1.5, label='Fronteira de Pareto')
plt.xlabel('Tempo de Treino (s)')
plt.ylabel('F1 Score')
plt.title('Trade-off: F1 vs Tempo de Treino')
plt.legend()
plt.grid(alpha=0.3)

# F1 vs Tamanho do Modelo
plt.subplot(1, 2, 2)
plt.scatter(df['model_size']/1e6, df['f1'], c='gray', alpha=0.4, s=30, label='Todas combinações')
plt.scatter(pareto['model_size']/1e6, pareto['f1'], c='red', s=80,
            edgecolors='black', linewidth=1.5, label='Fronteira de Pareto')
plt.xlabel('Tamanho do Modelo (MB)')
plt.ylabel('F1 Score')
plt.title('Trade-off: F1 vx Tamanho do Modelo')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

pareto_sorted = pareto.sort_values('f1', ascending=False)[[
    'combo_id', 'n_estimators', 'learning_rate', 'max_depth', 'subsample',
    'f1', 'runtime_train', 'runtime_inf', 'model_size'
]].copy()
pareto_sorted['model_size_mb'] = (pareto_sorted['model_size'] / 1e6).round(2)
pareto_sorted = pareto_sorted.drop(columns=['model_size'])
display(pareto_sorted.reset_index(drop=True))

best_f1 = pareto_sorted.iloc[0]
best_speed = pareto_sorted.loc[pareto_sorted['runtime_train'].idxmin()]
best_size = pareto_sorted.loc[pareto_sorted['model_size_mb'].idxmin()]

print(f"\nF1: combo_id={int(best_f1['combo_id'])} → F1={best_f1['f1']:.4f}")
print(f"menortempo: combo_id={int(best_speed['combo_id'])} → {best_speed['runtime_train']:.2f}s")
print(f"menor tamanho: combo_id={int(best_size['combo_id'])} → {best_size['model_size_mb']:.2f}MB")


#%% [Markdown] 
#
'''best_params_xgb = {
    'n_estimators': 300,    
    'learning_rate': 0.1,
    'max_depth': 9,
    'subsample': 1,
    'n_jobs': -1,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': ['logloss', 'error'] 
}'''
best_params_xgb = {
    'n_estimators': 220,    
    'learning_rate': 0.09,
    'max_depth': 10,
    'subsample': 0.9,
    'n_jobs': -1,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': ['logloss', 'error'] 
}
#%% [Markdown] 
# Curva
def plot_learning_curves(external_dir, params):
    
    folds = sorted([
        int(f.replace("STRESS_fold", "").replace("_train.csv", ""))
        for f in os.listdir(external_dir) 
        if f.startswith("STRESS_fold") and f.endswith("_train.csv")
    ])
    

    n_folds = len(folds)
    n_cols = 2
    n_rows = math.ceil(n_folds / n_cols)
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()
    
    for i, fold in enumerate(tqdm(folds, desc="Treinamento")):
        ax = axes[i]
        
        train_path = os.path.join(external_dir, f"STRESS_fold{fold}_train.csv")
        test_path = os.path.join(external_dir, f"STRESS_fold{fold}_test.csv")
        
        X_train, y_train = fold_loader(train_path)
        X_test, y_test = fold_loader(test_path)
        
        model = XGBClassifier(**params)
        
        model.fit(
            X_train, y_train, 
            eval_set=[(X_train, y_train), (X_test, y_test)], 
            verbose=False
        )
        
        results = model.evals_result()
        epochs = len(results['validation_0']['logloss'])
        x_axis = range(0, epochs)
        
        ax.plot(x_axis, results['validation_0']['logloss'], label='Treino', color='blue', linewidth=1.5)
        ax.plot(x_axis, results['validation_1']['logloss'], label='Teste (Validação)', color='orange', linewidth=2)
        
        ax.set_title(f"Fold {fold}", fontsize=12, fontweight='bold')
        ax.set_ylabel('Log Loss')
        ax.set_xlabel('Iterações (Árvores)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        min_loss_idx = results['validation_1']['logloss'].index(min(results['validation_1']['logloss']))
        min_loss_val = min(results['validation_1']['logloss'])
        ax.axvline(min_loss_idx, color='red', linestyle='--', alpha=0.5)
        ax.text(min_loss_idx, min_loss_val, f' Min: {min_loss_val:.3f}', color='red', fontsize=9)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
plot_learning_curves(EXTERNAL_FOLD_DIR, best_params_xgb)

# %%
df_summary = pd.read_csv(os.path.join(GRID_SEARCH_DIR, "xgb_grid_summary.csv"))

best_row = df_summary[df_summary['combo_id'] == 40].iloc[0]
best_params_xgb = {
    'n_estimators': int(best_row['n_estimators']),
    'learning_rate': float(best_row['learning_rate']),
    'max_depth': int(best_row['max_depth']),
    'subsample': float(best_row['subsample'])
}

print(best_params_xgb)
# %%
def plot_xgb_feature_importance(train_path, best_params, importance_type='gain', top_n=20, figsize=(10, 8)):
 
    X_train, y_train = fold_loader(train_path)
    
    params = best_params.copy()
    if 'n_estimators' in params: params['n_estimators'] = int(params['n_estimators'])
    if 'max_depth' in params: params['max_depth'] = int(params['max_depth'])
    
    # Remove conflicting parameters
    params.pop('n_jobs', None)
    params.pop('random_state', None)
    params.pop('use_label_encoder', None)
    params.pop('eval_metric', None)
    
    model = XGBClassifier(
        n_jobs=-1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        **params
    )
    model.fit(X_train, y_train)
    
    booster = model.get_booster()
    importance_dict = booster.get_score(importance_type=importance_type)
    
    feat_imp = pd.DataFrame({
        'feature': list(importance_dict.keys()),
        'importance': list(importance_dict.values())
    }).sort_values('importance', ascending=False)
    
    feat_imp['importance_normalized'] = feat_imp['importance'] / feat_imp['importance'].sum()
    
    top_features = feat_imp.head(top_n)
    
    plt.figure(figsize=figsize)
    sns.barplot(
        data=top_features,
        x='importance_normalized',
        y='feature',
        palette='rocket'
    )
    
    plt.title(f'Features mais importantes  XGBoost ({importance_type})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Importância Normalizada', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(top_features['importance_normalized']):
        plt.text(v + 0.005, i, f'{v:.1%}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return feat_imp[['feature', 'importance', 'importance_normalized']]

#%%
train_path = os.path.join(EXTERNAL_FOLD_DIR, "STRESS_fold1_train.csv")

best_params_xgb = {
    'n_estimators': 220,    
    'learning_rate': 0.09,
    'max_depth': 10,
    'subsample': 0.9,
    'n_jobs': -1,
    'random_state': 42,
    'use_label_encoder': False,
    'eval_metric': 'logloss'
}

feature_importance_xgb = plot_xgb_feature_importance(
    train_path=train_path,
    best_params=best_params_xgb,
    importance_type='gain', 
    top_n=20,
    figsize=(10, 8)
)

feature_importance_xgb.to_csv(
    os.path.join(GRID_SEARCH_DIR, "xgb_feature_importance.csv"),
    index=False
)
# %%
