# %% [Markdown] 
# Importações e Configurações Iniciais
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
    auc, precision_recall_curve, confusion_matrix, 
    ConfusionMatrixDisplay, RocCurveDisplay
)

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
# Definição do Grid de Hiperparâmetros
param_grid_xgb = {
    'n_estimators': [180, 220, 260, 300, 340],
    'learning_rate': [0.08, 0.09, 0.1, 0.11, 0.12],    
    'max_depth': [8, 9, 10],             
    'subsample': [0.9, 1.0]
}

# %% [Markdown] 
# Grid Search XGBoost
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
            
            # TAMANHO DO MODELO
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
        # Remove colunas não numéricas ou de ID do dicionário de média
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
    print(f"\nMelhor Combinação XGBoost (Combo ID {best_combo['combo_id']}):")
    print(best_combo[param_names])
    
    return best_combo[param_names].to_dict()

# Executa o Grid Search
best_params_xgb = grid_search_xgb(INTERNAL_FOLDS_DIR, param_grid_xgb, metric_sort="f1")

# %% [Markdown] 
# Avaliação Externa (Quick Check)
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

# %% [Markdown] 
# Análise de Pareto
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

# Visualização Pareto
plt.figure(figsize=(12, 5))

# F1 vs Tempo
plt.subplot(1, 2, 1)
plt.scatter(df['runtime_train'], df['f1'], c='gray', alpha=0.4, s=30, label='Todas combinações')
plt.scatter(pareto['runtime_train'], pareto['f1'], c='red', s=80, 
            edgecolors='black', linewidth=1.5, label='Fronteira de Pareto')
plt.xlabel('Tempo de Treino (s)')
plt.ylabel('F1 Score')
plt.title('XGBoost: F1 vs Tempo de Treino')
plt.legend()
plt.grid(alpha=0.3)

# F1 vs Tamanho
plt.subplot(1, 2, 2)
plt.scatter(df['model_size']/1e6, df['f1'], c='gray', alpha=0.4, s=30, label='Todas combinações')
plt.scatter(pareto['model_size']/1e6, pareto['f1'], c='red', s=80,
            edgecolors='black', linewidth=1.5, label='Fronteira de Pareto')
plt.xlabel('Tamanho do Modelo (MB)')
plt.ylabel('F1 Score')
plt.title('XGBoost: F1 vs Tamanho do Modelo')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.show()

# Tabela Pareto
pareto_sorted = pareto.sort_values('f1', ascending=False)[[
    'combo_id', 'n_estimators', 'learning_rate', 'max_depth', 'subsample',
    'f1', 'runtime_train', 'runtime_inf', 'model_size'
]].copy()
pareto_sorted['model_size_mb'] = (pareto_sorted['model_size'] / 1e6).round(2)
pareto_sorted = pareto_sorted.drop(columns=['model_size'])
display(pareto_sorted.reset_index(drop=True))

# %% [Markdown] 
# Curvas de Aprendizado (Learning Curves)
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
    
    for i, fold in enumerate(tqdm(folds, desc="Gerando Learning Curves")):
        ax = axes[i]
        
        train_path = os.path.join(external_dir, f"STRESS_fold{fold}_train.csv")
        test_path = os.path.join(external_dir, f"STRESS_fold{fold}_test.csv")
        
        X_train, y_train = fold_loader(train_path)
        X_test, y_test = fold_loader(test_path)
        
        # Copia params para não alterar o original
        current_params = params.copy()
        current_params.pop('n_jobs', None) # Remove n_jobs se existir para evitar warnings no fit interno
        
        model = XGBClassifier(**current_params)
        
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
        
        ax.set_title(f"Fold {fold} (XGBoost)", fontsize=12, fontweight='bold')
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

# %% [Markdown] 
# Feature Importance
def plot_xgb_feature_importance(train_path, best_params, importance_type='gain', top_n=20, figsize=(10, 8)):
    X_train, y_train = fold_loader(train_path)
    
    params = best_params.copy()
    # Converte tipos
    if 'n_estimators' in params: params['n_estimators'] = int(params['n_estimators'])
    if 'max_depth' in params: params['max_depth'] = int(params['max_depth'])
    
    # Limpeza de params conflitantes
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
    
    if not importance_dict:
        print("Nenhuma importância encontrada. Verifique se o modelo foi treinado corretamente.")
        return pd.DataFrame()

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
    
    plt.title(f'Features mais importantes - XGBoost ({importance_type})', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Importância Normalizada', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.grid(axis='x', alpha=0.3)
    
    for i, v in enumerate(top_features['importance_normalized']):
        plt.text(v + 0.005, i, f'{v:.1%}', va='center', fontsize=9)
    
    plt.tight_layout()
    plt.show()
    
    return feat_imp[['feature', 'importance', 'importance_normalized']]

train_path = os.path.join(EXTERNAL_FOLD_DIR, "STRESS_fold1_train.csv")
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

# %% [Markdown] 
# Avaliação Final (Matriz de Confusão e Curva ROC) com Melhor Modelo
# Carrega o resultado do grid para garantir que estamos usando o melhor modelo
df_summary_final = pd.read_csv(os.path.join(GRID_SEARCH_DIR, "xgb_grid_summary.csv"))
best_row_final = df_summary_final.loc[df_summary_final['f1'].idxmax()]

print(f"--- CONFIGURAÇÃO SELECIONADA PARA AVALIAÇÃO FINAL ---")
print(f"Combo ID: {int(best_row_final['combo_id'])}")
print(f"F1 Score: {best_row_final['f1']:.4f}")

xgb_params_final = {
    'n_estimators': int(best_row_final['n_estimators']),
    'learning_rate': float(best_row_final['learning_rate']),
    'max_depth': int(best_row_final['max_depth']),
    'subsample': float(best_row_final['subsample']),
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1
}

# Loop de avaliação final
folds = sorted([
    int(f.replace("STRESS_fold", "").replace("_train.csv", ""))
    for f in os.listdir(EXTERNAL_FOLD_DIR) 
    if f.startswith("STRESS_fold") and f.endswith("_train.csv")
])

all_y_test_xgb = []
all_y_pred_xgb = []
all_y_proba_xgb = []

for fold in tqdm(folds, desc="Avaliando XGBoost (Final)"):
    train_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_train.csv")
    test_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_test.csv")
    
    X_train, y_train = fold_loader(train_path)
    X_test, y_test = fold_loader(test_path)
    
    model = XGBClassifier(**xgb_params_final)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    all_y_test_xgb.extend(y_test)
    all_y_pred_xgb.extend(y_pred)
    all_y_proba_xgb.extend(y_proba)

# --- GRÁFICO 1: MATRIZ DE CONFUSÃO ---
fig, ax = plt.subplots(figsize=(6, 5))
cm_xgb = confusion_matrix(all_y_test_xgb, all_y_pred_xgb)

disp_xgb = ConfusionMatrixDisplay(
    confusion_matrix=cm_xgb, 
    display_labels=['Rest (0)', 'Stress (1)']
)

disp_xgb.plot(ax=ax, cmap='Greens', values_format='d')
plt.title('Matriz de Confusão - XGBoost', fontsize=14, fontweight='bold')
plt.grid(False)
plt.show()

# --- GRÁFICO 2: CURVA ROC ---
fig, ax = plt.subplots(figsize=(7, 6))

RocCurveDisplay.from_predictions(
    all_y_test_xgb, 
    all_y_proba_xgb, 
    ax=ax, 
    name="XGBoost",
    color="darkgreen", 
    lw=2
)

ax.plot([0, 1], [0, 1], "k--", lw=2, label="Aleatório (chance)")

ax.set_title("Curva ROC - XGBoost", fontsize=14, fontweight='bold')
ax.set_xlabel("Taxa de Falsos Positivos", fontsize=12)
ax.set_ylabel("Taxa de Verdadeiros Positivos", fontsize=12)
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

plt.show()

# %% Salvar melhor modelo XGBoost
MODEL_SAVE_PATH_XGB = os.path.join(BASE_DIR, "best_xgb_model.pkl")

# Treino final com a configuração selecionada (usando o fold de exemplo ou treino completo)
X_final, y_final = fold_loader(os.path.join(EXTERNAL_FOLD_DIR, "STRESS_fold1_train.csv"))

final_model_xgb = XGBClassifier(**xgb_params_final)
final_model_xgb.fit(X_final, y_final)

# Salvando via pickle
with open(MODEL_SAVE_PATH_XGB, 'wb') as f:
    pickle.dump(final_model_xgb, f)

print(f"Modelo XGBoost salvo com sucesso em: {MODEL_SAVE_PATH_XGB}")
#%%