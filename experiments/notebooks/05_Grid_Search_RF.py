# %%
import os
import io
import time
import pickle
import itertools
from pathlib import Path
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    auc, precision_recall_curve,
    log_loss,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay
)


BASE_DIR = Path(__file__).resolve().parent.parent
FOLDS_DIR = os.path.join(BASE_DIR, "folds")

EXTERNAL_FOLD_DIR = os.path.join(FOLDS_DIR, "external_folds")
INTERNAL_FOLDS_DIR = os.path.join(FOLDS_DIR, "internal_folds")
GRID_SEARCH_DIR = os.path.join(FOLDS_DIR, "grid_search_rf")

os.makedirs(GRID_SEARCH_DIR, exist_ok=True)

TARGET = "label"
base_seed = 42

LEARNING_CURVE_STEP = 5
FEATURE_IMPORTANCE_TOP_N = 20
FEATURE_LABEL_OFFSET = 0.001


# %% [Markdown] Carregamento do Fold
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

# %%[Markdown] Definição do Grid de Hiperparâmetros
param_grid = {
    'n_estimators': [150, 200, 250],
    'max_features': [0.6, 0.7, 0.8],
    'min_samples_leaf': [1, 2, 3]
}

# %%[Markdown] Grid Search RF
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
        
        if 'n_estimators' in params: params['n_estimators'] = int(params['n_estimators'])
        if 'min_samples_leaf' in params: params['min_samples_leaf'] = int(params['min_samples_leaf'])
        
        metrics_folds = []

        # TREINO E VALIDAÇÃO
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
    
    df_summary.to_csv(os.path.join(GRID_SEARCH_DIR, "grid_search_summary.csv"), index=False)
    df_folds.to_csv(os.path.join(GRID_SEARCH_DIR, "grid_search_folds_detail.csv"), index=False)
    
    df_summary = df_summary.sort_values(by=metric_sort, ascending=False)
    

    display(df_summary)
    
    best_combo = df_summary.iloc[0]
 
    print(best_combo[param_names])
    
    return best_combo[param_names].to_dict()

best_params = grid_search(INTERNAL_FOLDS_DIR, param_grid, metric_sort="f1")



# %%[Markdown] Avaliação do Random Forest
def evaluate_model(EXTERNAL_FOLD_DIR, best_params):
    
    folds = sorted([
        int(f.replace("STRESS_fold", "").replace("_train.csv", ""))
        for f in os.listdir(EXTERNAL_FOLD_DIR) 
        if f.startswith("STRESS_fold") and f.endswith("_train.csv")
    ])
    
    print(f"Folds Externos Identificados: {folds}")
    
    rows_eval = []
    
    # Ajuste dos tipos
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

    df_summary = df_results.agg(['mean', 'std'])
    df_summary.to_csv(os.path.join(GRID_SEARCH_DIR, "Evaluation_results_summary.csv"))
    display(df_summary)

evaluate_model(EXTERNAL_FOLD_DIR, best_params)


# %% [Markdown] Meslhores Parametros
best_params_rf = {
    'n_estimators': 250,      
    'max_features': 0.6,
    'min_samples_leaf': 1,
    'n_jobs': -1,
    'random_state': 42
}
#%% Curvas de Aprendizado
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
    
    step_size = LEARNING_CURVE_STEP

    max_trees = params['n_estimators']
    
    rf_params = params.copy()
    
    if 'n_estimators' in rf_params:
        del rf_params['n_estimators']
    
    if 'random_state' in rf_params:
        del rf_params['random_state']

    for i, fold in enumerate(tqdm(folds, desc="Processando Folds")):
        ax = axes[i]
        
        train_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_train.csv")
        test_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_test.csv")
        
        X_train, y_train = fold_loader(train_path)
        X_test, y_test = fold_loader(test_path)
        
        train_loss = []
        test_loss = []
        n_trees_axis = []
        
        for n_trees in range(step_size, max_trees + 1, step_size):
            
            model = RandomForestClassifier(
                n_estimators=n_trees, 
                random_state=base_seed,  
                **rf_params
            )
        
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

#%% Importancia das Features
def plot_rf_feature_importance(fold_dir, best_params, top_n=FEATURE_IMPORTANCE_TOP_N, figsize=(10, 8)):
    params = best_params.copy()
    params.pop('n_jobs', None)

    fold_ids = sorted([
        int(f.replace("STRESS_fold", "").replace("_train.csv", ""))
        for f in os.listdir(fold_dir)
        if f.startswith("STRESS_fold") and f.endswith("_train.csv")
    ])

    all_importances = []
    feature_names = None

    for fold_id in fold_ids:
        X_train, y_train = fold_loader(os.path.join(fold_dir, f"STRESS_fold{fold_id}_train.csv"))
        model = RandomForestClassifier(n_jobs=-1, **params)
        model.fit(X_train, y_train)
        all_importances.append(model.feature_importances_)
        if feature_names is None:
            feature_names = X_train.columns

    importances_arr = np.array(all_importances)
    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': importances_arr.mean(axis=0),
        'importance_std': importances_arr.std(axis=0)
    }).sort_values('importance_mean', ascending=False)

    top_features = feat_imp.head(top_n)

    plt.figure(figsize=figsize)
    sns.barplot(
        data=top_features,
        x='importance_mean',
        y='feature',
        palette='viridis'
    )

    plt.title('Features mais importantes para Random Forest\n(Média ± std de todos os folds)', fontsize=14, fontweight='bold')
    plt.xlabel('Importância Média (Mean Decrease Impurity)', fontsize=11)
    plt.ylabel('Feature', fontsize=11)
    plt.grid(axis='x', alpha=0.3)

    for i, (v, s) in enumerate(zip(top_features['importance_mean'], top_features['importance_std'])):
        plt.text(v + FEATURE_LABEL_OFFSET, i, f'{v:.4f}±{s:.4f}', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

    return feat_imp

# %%
feature_importance_df = plot_rf_feature_importance(
    fold_dir=EXTERNAL_FOLD_DIR,
    best_params=best_params_rf,
    top_n=FEATURE_IMPORTANCE_TOP_N,
    figsize=(10, 8)
)

feature_importance_df.to_csv(
    os.path.join(GRID_SEARCH_DIR, "rf_feature_importance.csv"),
    index=False
)

# %% Matriz de Confusão e Curva ROC
rf_params = {
    'n_estimators': 250,      
    'max_features': 0.6,
    #'max_depth': 15,          
    'min_samples_leaf': 1,   
    'n_jobs': -1,             
    'random_state': 42
}

folds = sorted([
    int(f.replace("STRESS_fold", "").replace("_train.csv", ""))
    for f in os.listdir(EXTERNAL_FOLD_DIR) 
    if f.startswith("STRESS_fold") and f.endswith("_train.csv")
])

print(f"Total de Folds: {len(folds)}")

all_y_test_rf = []
all_y_pred_rf = []
all_y_proba_rf = []

for fold in tqdm(folds, desc="Avaliando Random Forest nos Folds"):
    train_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_train.csv")
    test_path = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold}_test.csv")
    
    X_train, y_train = fold_loader(train_path)
    X_test, y_test = fold_loader(test_path)
    
    model = RandomForestClassifier(**rf_params)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    all_y_test_rf.extend(y_test)
    all_y_pred_rf.extend(y_pred)
    all_y_proba_rf.extend(y_proba)

fig, ax = plt.subplots(figsize=(6, 5))
cm_rf = confusion_matrix(all_y_test_rf, all_y_pred_rf)
disp_rf = ConfusionMatrixDisplay(
    confusion_matrix=cm_rf, 
    display_labels=['Rest (0)', 'Stress (1)']
)

disp_rf.plot(
    ax=ax, 
    cmap='Purples',
    values_format='d'
)

plt.title('Matriz de Confusão - Random Forest ', fontsize=14, fontweight='bold')
plt.grid(False)
plt.show()

fig, ax = plt.subplots(figsize=(7, 6))

RocCurveDisplay.from_predictions(
    all_y_test_rf, 
    all_y_proba_rf, 
    ax=ax, 
    name="Random Forest",
    color="purple", 
    lw=2
)

ax.plot([0, 1], [0, 1], "k--", lw=2, label="Aleatório (chance)")

ax.set_title("Curva ROC - Random Forest (Total dos Folds)", fontsize=14, fontweight='bold')
ax.set_xlabel("Taxa de Falsos Positivos", fontsize=12)
ax.set_ylabel("Taxa de Verdadeiros Positivos", fontsize=12)
ax.legend(loc="lower right")
ax.grid(alpha=0.3)

plt.show()


# %%
# %% [Markdown] Salva o Modelo
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "best_rf_model.pkl")

# Train final model on all external fold training sets combined
all_X_final, all_y_final = [], []
for fold_id in folds:
    X_fold, y_fold = fold_loader(os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{fold_id}_train.csv"))
    all_X_final.append(X_fold)
    all_y_final.append(y_fold)
X_final = pd.concat(all_X_final, ignore_index=True)
y_final = pd.concat(all_y_final, ignore_index=True)

final_model = RandomForestClassifier(**best_params_rf)
final_model.fit(X_final, y_final)

with open(MODEL_SAVE_PATH, 'wb') as f:
    pickle.dump(final_model, f)
# %%
