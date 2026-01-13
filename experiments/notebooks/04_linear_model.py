#%% Importações

# ---------------------------
# Manipulação de dados
# ---------------------------
import collections
import numpy as np
import os
import pandas as pd
import statistics
import sys
import time
import timeit
import math

from pathlib import Path
sys.path.insert(0, str(Path.cwd().parent.parent))
from src.config import RANDOM_SEED


# ---------------------------
# Estatística e testes estatísticos
# ---------------------------
import statsmodels.api as sm

from scipy import stats
from scipy.stats import (
    anderson, chi2_contingency, f_oneway, kstest, normaltest,
    shapiro, ttest_rel, wilcoxon, mannwhitneyu
)
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ---------------------------
# Salvamento e análise de tamanho do modelo
# ---------------------------
import joblib
from pympler import asizeof

# ---------------------------
# Pré-processamento
# ---------------------------
import missingno as msno
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# ---------------------------
# Divisão do dataset
# ---------------------------
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold

# ---------------------------
# Parametrização
# ---------------------------
from sklearn.model_selection import GridSearchCV

# ---------------------------
# Modelos de aprendizado de máquina
# ---------------------------
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression



# ---------------------------
# Métricas de avaliação
# ---------------------------
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, mean_squared_error,
    precision_score, r2_score, recall_score, silhouette_score,
    mean_absolute_error, mean_absolute_percentage_error,
    median_absolute_error, max_error,
    roc_auc_score, average_precision_score, log_loss,
    matthews_corrcoef, cohen_kappa_score, brier_score_loss,
    hamming_loss, jaccard_score, fbeta_score,
    balanced_accuracy_score, zero_one_loss,
    silhouette_score, calinski_harabasz_score, davies_bouldin_score
)

# ---------------------------
# Dimensionalidade
# ---------------------------
from sklearn.decomposition import PCA

# ---------------------------
# Visualização
# ---------------------------
import matplotlib.pyplot as plt
import plotly.express as px
import seaborn as sns
import os
from pathlib import Path

# Cores ANSI
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

#%%[markdown]
# CARREGANDO O DATASET
current_dir = Path(os.getcwd())
processed_dir = current_dir / 'experiments' / 'processed'
print(processed_dir)

data_dir = Path.cwd().parent / 'processed'
df_exercise = pd.read_csv(data_dir / 'dataset_exercise.csv')
df_stress = pd.read_csv(data_dir / 'dataset_stress.csv')

#%%
df_stress.columns

#%%
#### Roda o OLS

features_significativas = [
    'acc_x_mean', 'acc_x_std', 'acc_y_std', 
    'acc_mean', 'acc_std', 'acc_ratio_up', 'acc_ratio_down',
    'mean_raw_eda', 'std_raw_eda', 
    'mean_tonic_eda', 'std_tonic_eda', 'tonic_ratio_up', 'tonic_ratio_down',
    'mean_phasic_eda', 'std_phasic_eda', 
    'scr_mean_amp', 'scr_mean_height', 
    'hr_mean', 
    'max_ibi', 'min_ibi', 'mean_ibi', 
    'pnn50', 'rmssd', 'sdnn', 
    'total_power', 'ratio', 
    'VLF_power', 'LF_power', 'VHF_power', 'VHF_peak'
]

X = df_stress[features_significativas]
X = sm.add_constant(X)

y = df_stress['label']

model = sm.OLS(y, X).fit()
print(model.summary())
# %%

def plot_residuos(model, titulo='Resíduo vs Valor Predito', usar_lowess=True):

    residuals = model.resid
    predicted_values = model.fittedvalues

    plt.figure(figsize=(8, 6))
    plt.scatter(predicted_values, residuals, color='blue', alpha=0.5)
    plt.axhline(y=0, color='red', linestyle='--')


    if usar_lowess:
        try:
            sns.residplot(x=predicted_values, y=residuals, lowess=True,
                          line_kws={'color': 'red', 'lw': 2, 'linestyle': '--'})
        except Exception as e:
            print("Erro ao aplicar LOWESS (linha suavizada):", e)

    plt.title(titulo)
    plt.xlabel('Valor predito')
    plt.ylabel('Resíduo')
    plt.show()
# %%
plot_residuos(model, titulo='Análise dos Resíduos do Modelo Linear')


# %%
def avaliar_desempenho(y_test, y_pred, n_features):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    
    # PROTEÇÃO CONTRA DIVISÃO POR ZERO NO MAPE
    try:
        # O sklearn pode gerar erro ou warning gigantesco aqui se y_test tiver zeros
        mape = mean_absolute_percentage_error(y_test, y_pred)
    except:
        mape = np.nan # Define como nulo se falhar
        
    mdae = median_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    me = max_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    n = len(y_test)
    p = n_features
    
    # Proteção para R2 ajustado se n for pequeno demais
    if (n - p - 1) != 0:
        r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1)
    else:
        r2_adj = np.nan
        
    # EVS
    var_test = np.var(y_test)
    if var_test != 0:
        evs = 1 - np.var(y_test - y_pred) / var_test
    else:
        evs = 0.0 # Se a variância for 0, o score é irrelevante

    return {
        "mse": mse,
        "mae": mae,
        "mape": mape,  # Provavelmente será inf ou gigante
        "mdae": mdae,
        "rmse": rmse,
        "me": me,
        "r2": r2,
        "r2_adj": r2_adj,
        "evs": evs
    }
# %%
X = df_stress.drop(columns=['protocol', 'label'])
X
#%%
y = df_stress['label']
y
# %%
kf = KFold(n_splits=10, shuffle=True, random_state=RANDOM_SEED)
kf

# %%
resultados = [] # lista para armazenar as métricas
modelos_por_fold = [] # lista para armazenar os modelos


for fold, (train, test) in enumerate(kf.split(X), 1):
    X_train, X_test = X.iloc[train], X.iloc[test]
    y_train, y_test = y.iloc[train], y.iloc[test]

    feature_cols = X_train.select_dtypes(include=[np.number]).columns
    X_train = X_train[feature_cols]
    X_test = X_test[feature_cols]

    modelo = LinearRegression()

    # Treinamento
    start_train = timeit.default_timer()
    modelo.fit(X_train, y_train)
    end_train = timeit.default_timer()
    train_time = end_train - start_train

    # Predição
    start_pred = timeit.default_timer()
    y_pred = modelo.predict(X_test)
    end_pred = timeit.default_timer()
    pred_time = end_pred - start_pred

    # Avaliação
    met = avaliar_desempenho(y_test, y_pred, X_test.shape[1])

    # Tamanho do modelo
    filename = f'modelo_fold{fold}.joblib'
    joblib.dump(modelo, filename)
    model_size_mb = os.path.getsize(filename) / (1024 * 1024)
    os.remove(filename)

    # Salva resultados
    resultados.append({
        "fold": fold,
        "train_time_s": train_time,
        "pred_time_s": pred_time,
        "model_size_mb": model_size_mb,
        **met
    })

    # Salva o modelo em memória
    modelos_por_fold.append(modelo)
# %%
resultados_lr = pd.DataFrame(resultados)
resultados_lr
# %%
print(f"\nMédias dos folds:")
print(resultados_lr.mean(numeric_only=True).to_string(float_format='%.4f'))
# %%

def validar_modelo_regressao(modelo, X, y, kfold, func_avaliacao, salvar_modelos=False):
    """
    Executa validação cruzada com k folds para um modelo de regressão qualquer,
    medindo tempo de treino, predição, tamanho do modelo e métricas.

    Parâmetros:
    - modelo: objeto sklearn-like com fit(X_train, y_train) e predict(X_test)
    - X: pandas DataFrame com features
    - y: pandas Series com target
    - kfold: objeto KFold já configurado (ex: KFold(n_splits=10, shuffle=True))
    - func_avaliacao: função que recebe (y_test, y_pred, n_features) e retorna dict métricas
    - salvar_modelos: bool, se True salva os arquivos dos modelos (pasta atual)

    Retorna:
    - DataFrame com resultados por fold
    """
    resultados = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        modelo_fold = modelo.__class__(**modelo.get_params())  # cria nova instância do mesmo modelo

        start_train = timeit.default_timer()
        modelo_fold.fit(X_train, y_train)
        end_train = timeit.default_timer()
        train_time = end_train - start_train

        start_pred = timeit.default_timer()
        y_pred = modelo_fold.predict(X_test)
        end_pred = timeit.default_timer()
        pred_time = end_pred - start_pred

        met = func_avaliacao(y_test, y_pred, X_test.shape[1])

        filename = f'modelo_fold{fold}.joblib'
        joblib.dump(modelo_fold, filename)
        model_size_mb = os.path.getsize(filename) / (1024 * 1024)
        if not salvar_modelos:
            os.remove(filename)

        resultados.append({
            "fold": fold,
            "train_time_s": train_time,
            "pred_time_s": pred_time,
            "model_size_mb": model_size_mb,
            **met
        })

    return pd.DataFrame(resultados)

# %%


le = LabelEncoder()
y = le.fit_transform(df_stress['label'])
print(f"\nMapeamento: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Remover colunas não-numéricas de X
X = df_stress.drop(columns=['label', 'subject_id', 'window_id'], errors='ignore')
X = X.select_dtypes(include=[np.number])

# Agora usar CLASSIFICADOR ao invés de regressor
from sklearn.ensemble import RandomForestClassifier

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

print(f"\nMelhores parâmetros: {grid_search.best_params_}")
print(f"Melhor score: {grid_search.best_score_:.4f}")
# %%
best_rf = grid_search.best_estimator_

y = pd.Series(y, index=df_stress.index, name='label')
resultados_rf = validar_modelo_regressao(best_rf, X, y, kf, avaliar_desempenho)
resultados_rf

# %%
print(f"\nMédias dos folds:")
print(resultados_rf.mean(numeric_only=True).to_string(float_format='%.4f'))
# %%
# %%
# Salvar o melhor modelo Random Forest treinado
model_path = processed_dir /  'best_random_forest_model.joblib'
model_path.parent.mkdir(parents=True, exist_ok=True)  # Criar pasta se não existir
joblib.dump(best_rf, model_path)
print(f"Modelo salvo em: {model_path}")
print(f"Tamanho do arquivo: {os.path.getsize(model_path) / (1024 * 1024):.2f} MB")

# %%
# Balanceamento de classes
# ---------------------------
from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.feature_selection import SelectKBest, f_classif

# %%
def comparar_balanceadores(titulo, df, random_state):
    # Preparar dados
    X = df.drop(columns=['label', 'subject_id', 'window_id'], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df['label']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Feature Selection
    selector = SelectKBest(f_classif, k=13)
    X_train_sel = selector.fit_transform(X_train, y_train)
    X_test_sel = selector.transform(X_test)
    
    # Balanceadores
    balanceadores = {
        "Sem Balanceamento": None,
        "SMOTE": SMOTE(random_state=random_state),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=random_state),
        "SVMSMOTE": SVMSMOTE(random_state=random_state),
        "ADASYN": ADASYN(random_state=random_state, n_neighbors=3),
        "RandomOverSampler": RandomOverSampler(random_state=random_state),
        "RandomUnderSampler": RandomUnderSampler(random_state=random_state),
        "TomekLinks": TomekLinks(),
        "NearMiss": NearMiss(version=1),
        "SMOTEENN": SMOTEENN(random_state=random_state),
        "SMOTETomek": SMOTETomek(random_state=random_state)
    }
    
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    resultados = []
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    for nome_bal, bal in balanceadores.items():
        try:
            # Aplica o balanceador
            if bal is None:
                X_res, y_res = X_train_sel, y_train
            else:
                X_res, y_res = bal.fit_resample(X_train_sel, y_train)
            
            # Cross Validation
            scores = cross_validate(model, X_res, y_res, cv=cv, scoring='accuracy')
            
            resultados.append({
                'Balanceador': nome_bal,
                'Acurácia Média': scores['test_score'].mean(),
                'Std': scores['test_score'].std(),
                'Amostras Treino': len(y_res)
            })
            
        except ValueError as e:
            print(f"⚠️ {nome_bal}: {str(e)}")
            resultados.append({
                'Balanceador': nome_bal,
                'Acurácia Média': np.nan,
                'Std': np.nan,
                'Amostras Treino': np.nan
            })
    
    return pd.DataFrame(resultados).sort_values('Acurácia Média', ascending=False)
