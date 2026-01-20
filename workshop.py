#%%
import os
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

#%%[Markdown]
# Caminhos
CSV_PATH = os.path.join("experiments", "processed", "dataset_stress.csv")
BASE_OUTPUT_DIR = "experiments"
FOLDS_DIR = os.path.join(BASE_OUTPUT_DIR, "folds")
INTERNAL_FOLDS_DIR = os.path.join(FOLDS_DIR, "internal")
EXTERNAL_REGRESSION_DIR = os.path.join(BASE_OUTPUT_DIR, "external_folds")

#%%
target = "label"



# %%
data = pd.read_csv(csv_path)
display(data.shape)
# %%
display(data.columns)
# %%
# coluna alvo - terceira coluna coluna
y = data.iloc[:, 3]
display(y)

# %%
x = data.drop(columns=['subject_id', 'window_id', 'label', 'scenario', 'protocol'])
display(x)


#%%
# distribuição dos indices conforme o STRATIFIED K FOLD

skf = StratifiedKFold(
    n_splits     = 5,
    shuffle      = True,
    random_state = 42
)

#%% Salva apenas 1 fold
folds = []

for fold_number, (train_index, test_index) in enumerate(skf.split(x, y), start=1):
    
    folds.append({
            "fold":fold_number,
            "train_index":train_index,
            "test_index":test_index
    })
display(folds)
# %%
fold0 = folds[0]
display(fold0)

idx_train = fold0["train_index"]
idx_test = fold0["test_index"]

#%%
# qtde total de indices

type(idx_train)
len(idx_train)
type(idx_test)
len(idx_test)

len(idx_test) + len(idx_train)
# %%
# pegar os índices e seleciona-los do dataframe original

train = data.iloc[idx_train].reset_index(drop=True)
test = data.iloc[idx_test].reset_index(drop=True)
display(train)
display(test)

#%%

train_path = os.path.join("experiments", "folds", "fold0_train.csv")
test_path = os.path.join("experiments", "folds", "fold0_test.csv")

train.to_csv(train_path, index=False)
test.to_csv(test_path, index=False)
# %%[Markdown]
# informações

num_instances_train = train.shape[0]
num_instances_test = test.shape[0]
class_distribution_train = train[target].value_counts(normalize=True)
class_distribution_test = test[target].value_counts(normalize=True)


#%% Salva todos folds
info_folds = []

for fold_number, (train_index, test_index) in enumerate(skf.split(x, y), start=1):
    
    train = data.iloc[train_index].reset_index(drop=True)
    test = data.iloc[test_index].reset_index(drop=True)
    
    train_filename = f"fold{fold_number}_train.csv"
    test_filename = f"fold{fold_number}_test.csv"
    
    train_path = os.path.join("experiments", "folds", train_filename)
    test_path = os.path.join("experiments", "folds", test_filename)
    
    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    
    
    train_dist = train[target].value_counts(normalize=True).to_dict()
    test_dist = test[target].value_counts(normalize=True).to_dict()
    
    stats = {
        "fold": fold_number,
        "train_rows": train.shape[0],
        "test_rows": test.shape[0],
        "train_file": train_filename,
        "class_dist_train": train_dist,
        "class_dist_test": test_dist,
        "index_train": train_index.tolist(),
        "index_test": test_index.tolist()

    }
    info_folds.append(stats)

#%%[Markdown]

df_folds = pd.DataFrame(info_folds)
display(df_folds)


# %%


fold_train1 = pd.read_csv(r"experiments\folds\fold1_train.csv")
display(fold_train1.shape)

#%%
fold_train1["protocol"].isna().any()
display(fold_train1[fold_train1["protocol"].isna()])

# %%
train, validation = train_test_split(
                fold_train1,
                test_size    = 0.2,
                shuffle      = True,
                stratify     = fold_train1[target],
                random_state = 42
)
train.shape
test.shape
# %%
info_internal_folds = []
folderFolds = r"experiments\folds"
folderInternal = r"experiments\folds\internal"


os.makedirs(folderInternal, exist_ok=True)

# %%
for i in tqdm(range(1,6) ,
              desc = "Processando Folds"):

    train_path = os.path.join(folderFolds, f"fold{i}_train.csv")
    fold_train = pd.read_csv(train_path)
    
    
    train, validation = train_test_split(fold_train, 
                                     test_size=0.2,
                                     shuffle=True, 
                                     stratify=fold_train[target],
                                     random_state=42)
    
    internal_path_train      = os.path.join(folderInternal, f"train_{i}.csv")
    internal_path_validation = os.path.join(folderInternal, f"val_{i}.csv")

    train.to_csv(internal_path_train, index=False)
    validation.to_csv(internal_path_validation, index=False)

    info_fold = {
        "fold": i, 
        "num_instances_train": train.shape[0],
        "num_instances_test": test.shape[0],
        "class_distribution_train":(
            train[target].value_counts(normalize=True).to_dict()            
        ),
        "class_distribution_test":(
            test[target].value_counts(normalize=True).to_dict()            
        ),
        "train_index": train_index.tolist(),
        "test_index": test_index.tolist(),
    }

    info_internal_folds.append(info_fold)


df_info_folds = pd.DataFrame(info_internal_folds)
caminho_info = os.path.join(folderInternal, "info_folds.csv")
df_info_folds.to_csv(caminho_info, index= False)


# %% [Markdown]
# Regressão

folderRoot = output_dir_external
os.makedirs(folderRoot, exist_ok=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)                      
folds = []
for fold_number, (idx_train, idx_test) in enumerate(kf.split(x), start=1):
    folds.append({
            "fold"      : fold_number,
            "train_idx" : idx_train,
            "test_idx"  : idx_test
    })


# %%
info_folds = []

for i, fold in enumerate(tqdm(folds, desc="process"), start=1):
    
    train_index = fold["train_idx"]
    test_index = fold["test_idx"]    
    
    treino = data.iloc[train_index].reset_index(drop=True)
    teste = data.iloc[test_index].reset_index(drop=True)    
    
    # salvar arquivos
    caminho_treino = os.path.join(folderRoot, f"train_{i}.csv")
    caminho_teste = os.path.join(folderRoot, f"test_{i}.csv")

    treino.to_csv(caminho_treino, index = False)
    teste.to_csv(caminho_teste, index = False)

    info_fold = {
        "fold": i, 
        "num_instances_train": treino.shape[0],
        "num_instances_test": teste.shape[0],        
        "train_index": train_index.tolist(),
        "test_index": test_index.tolist(),  
    }

    info_folds.append(info_fold)

df_info_folds = pd.DataFrame(info_folds)
df_info_folds.to_csv(
    os.path.join(folderRoot, "info_folds.csv"),
    index=False
)
# %% [Markdown]

# Agrupamento Kfold
data_cluster = data.drop(columns=[target])
kf = KFold(n_splits=5, shuffle=True, random_state=42)

folds = []
for fold_number, (train_idx, test_idx) in enumerate(kf.split(data_cluster), start=1):   
     folds.append(
        {
            "fold": fold_number,
            "train_idx": train_idx,
            "test_idx": test_idx,
        }
    )

folds
# %%  [Markdown]
# Agrupamento Kfold