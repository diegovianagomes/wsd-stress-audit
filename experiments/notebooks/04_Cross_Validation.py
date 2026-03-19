#%%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold, GroupKFold, StratifiedKFold,LeaveOneGroupOut, train_test_split

#%%
print("Diretório atual:", os.getcwd())
#%%[Markdown]
# Caminhos
CSV_PATH = os.path.join("..", "processed", "dataset_stress.csv")
BASE_OUTPUT_DIR = ".."
FOLDS_DIR = os.path.join(BASE_OUTPUT_DIR, "folds")
INTERNAL_FOLDS_DIR = os.path.join(FOLDS_DIR, "internal_folds")
EXTERNAL_FOLD_DIR = os.path.join(FOLDS_DIR, "external_folds")
KFOLD_DIR = os.path.join(FOLDS_DIR, "kfold_regression")
CLUSTER_DIR = os.path.join(FOLDS_DIR, "kfold_cluster")
LOGO_DIR = os.path.join(FOLDS_DIR, "logo_fold")


#%%[Markdown]
# Variavel Alvo
TARGET = "label"
GROUP_COL = "subject_id"

#%%[Markdown}
# Cria os diretórios 
os.makedirs(FOLDS_DIR, exist_ok=True)
os.makedirs(INTERNAL_FOLDS_DIR, exist_ok=True)
os.makedirs(EXTERNAL_FOLD_DIR, exist_ok=True)
os.makedirs(KFOLD_DIR, exist_ok=True)
os.makedirs(CLUSTER_DIR, exist_ok=True)
os.makedirs(LOGO_DIR, exist_ok=True)

# %% [Markdown] 
#
def carregar_dados(caminho):
    data = pd.read_csv(caminho)
    
    X = data.drop(columns=['subject_id', 'window_id', 'label', 'scenario', 'protocol'])
    y = data[TARGET]
    return data, X, y

data_full, X_data, y_data = carregar_dados(CSV_PATH)

# %% [Markdown] 
# Folds externos
def external_folds(data, X, y, n_splits=10):
    
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    info_folds = []

    for fold_number, (train_index, test_index) in enumerate(tqdm(skf.split(X, y), total=n_splits, desc="External Folds"), start=1):
        
        train = data.iloc[train_index].reset_index(drop=True)
        test = data.iloc[test_index].reset_index(drop=True)
        
        train_filename = f"STRESS_fold{fold_number}_train.csv"
        test_filename = f"STRESS_fold{fold_number}_test.csv"
        
        train.to_csv(os.path.join(EXTERNAL_FOLD_DIR, train_filename), index=False)
        test.to_csv(os.path.join(EXTERNAL_FOLD_DIR, test_filename), index=False)
        
        stats = {
            "fold": fold_number,
            "train_rows": train.shape[0],
            "test_rows": test.shape[0],
            "train_file": train_filename,
            "class_dist_train": train[TARGET].value_counts(normalize=True).to_dict(),
            "class_dist_test": test[TARGET].value_counts(normalize=True).to_dict(),
            "train_index": train_index.tolist(),
            "test_index": test_index.tolist()
        }
        info_folds.append(stats)

    pd.DataFrame(info_folds).to_csv(os.path.join(EXTERNAL_FOLD_DIR, "STRESS_fold_info.csv"), index=False)

external_folds(data_full, X_data, y_data)

# %% [Markdown]
# Fold Inteno
def internal_folds(n_splits=10):
    
    info_internal = []

    for i in tqdm(range(1, n_splits + 1), desc="Internal Folds"):
        train_path_orig = os.path.join(EXTERNAL_FOLD_DIR, f"STRESS_fold{i}_train.csv")
        fold_train = pd.read_csv(train_path_orig)
        
        train, validation = train_test_split(
            fold_train,
            test_size=0.2,
            shuffle=True,
            stratify=fold_train[TARGET],
            random_state=42
        )
        
        train.to_csv(os.path.join(INTERNAL_FOLDS_DIR, f"STRESS_train_{i}.csv"), index=False)
        validation.to_csv(os.path.join(INTERNAL_FOLDS_DIR, f"STRESS_val_{i}.csv"), index=False)
        
        info_internal.append({
            "fold_origin": i,
            "internal_train_rows": train.shape[0],
            "internal_val_rows": validation.shape[0],
            "class_dist_train": train[TARGET].value_counts(normalize=True).to_dict(),
            "class_dist_val": validation[TARGET].value_counts(normalize=True).to_dict()
        })

    pd.DataFrame(info_internal).to_csv(os.path.join(INTERNAL_FOLDS_DIR, "STRESS_fold_info.csv"), index=False)

internal_folds()
# %% [Markdown]
# Kfold
# Gera folds para Regressão/Agrupamento 
def kfold(data, X, n_splits=10):
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    info_folds = []

    for fold_number, (train_index, test_index) in enumerate(tqdm(kf.split(X), total=n_splits, desc="Kfold"), start=1):
        
        train = data.iloc[train_index].reset_index(drop=True)
        test = data.iloc[test_index].reset_index(drop=True)
        
        train.to_csv(os.path.join(KFOLD_DIR, f"STRESS_train_{fold_number}.csv"), index=False)
        test.to_csv(os.path.join(KFOLD_DIR, f"STRESS_test_{fold_number}.csv"), index=False)
        
        info_folds.append({
            "fold": fold_number,
            "train_rows": train.shape[0],
            "test_rows": test.shape[0],
            "train_index": train_index.tolist(),
            "test_index": test_index.tolist()
        })

    pd.DataFrame(info_folds).to_csv(os.path.join(KFOLD_DIR, "STRESS_fold_info.csv"), index=False)

kfold(data_full, X_data)

# %% Agrupamento 
def kfold_cluster(data, target_col, group_col, n_splits=10):

    data_cluster = data.drop(columns=[target_col], errors='ignore')
    groups = data[group_col] 
    
    gkf = GroupKFold(n_splits=n_splits)
    info_folds = []

    for fold_number, (train_index, test_index) in enumerate(tqdm(gkf.split(data_cluster, groups=groups), total=n_splits, desc="Group KFold"), start=1):
        
        train = data_cluster.iloc[train_index].reset_index(drop=True)
        test = data_cluster.iloc[test_index].reset_index(drop=True)
        
        train.to_csv(os.path.join(CLUSTER_DIR, f"STRESS_train_group_{fold_number}.csv"), index=False)
        test.to_csv(os.path.join(CLUSTER_DIR, f"STRESS_test_group_{fold_number}.csv"), index=False)
        

        test_subjects = data.iloc[test_index][group_col].unique().tolist()
        
        info_folds.append({
            "fold": fold_number,
            "train_rows": train.shape[0],
            "test_rows": test.shape[0],
            "test_subjects": test_subjects, 
            "train_index": train_index.tolist(),
            "test_index": test_index.tolist()
        })

    pd.DataFrame(info_folds).to_csv(os.path.join(CLUSTER_DIR, "STRESS_info_cluster.csv"), index=False)

kfold_cluster(data_full, target_col=TARGET, group_col=GROUP_COL, n_splits=10)

# %% [Markdown]
# LOGO - LeaveOneGroupOut
def logo_folds(data, target_col, group_col='subject_id'):

    # Preparação
    X = data.drop(columns=[target_col], errors='ignore')
    y = data[target_col]
    groups = data[group_col]
    
    loo = LeaveOneGroupOut()
    info_folds = []
    
    total_subjects = data[group_col].nunique()

    for fold_number, (train_index, test_index) in enumerate(tqdm(loo.split(X, y, groups=groups), total=total_subjects, desc="LOGO Folds"), start=1):
        
        train = data.iloc[train_index].reset_index(drop=True)
        test = data.iloc[test_index].reset_index(drop=True)
        
        subject_test = test[group_col].iloc[0]
        
        train_filename = f"STRESS_train_{subject_test}.csv"
        test_filename = f"STRESS_test_{subject_test}.csv"
        
        train.to_csv(os.path.join(LOGO_DIR, train_filename), index=False)
        test.to_csv(os.path.join(LOGO_DIR, test_filename), index=False)
        
        info_folds.append({
            "fold": fold_number,
            "object_test": subject_test,
            "train_rows": train.shape[0],
            "test_rows": test.shape[0],
            "class_dist_test": test[target_col].value_counts(normalize=True).to_dict()
        })

    pd.DataFrame(info_folds).to_csv(os.path.join(LOGO_DIR, "STRESS_folds_info.csv"), index=False)

column_group = 'subject_id' if 'subject_id' in data_full.columns else 'participant'
logo_folds(data_full, target_col=TARGET, group_col=column_group)



# %%
