#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
import joblib

# Sklearn & Imblearn
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.preprocessing import (
    LabelEncoder, OneHotEncoder, RobustScaler, StandardScaler
)
from sklearn.feature_selection import RFE, SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import SMOTE, ADASYN, RandomOverSampler, BorderlineSMOTE, SVMSMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from imblearn.combine import SMOTEENN, SMOTETomek

# Modelos
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')



data_dir = Path.cwd().parent / 'processed' 

df_stress = pd.read_csv(data_dir / 'dataset_stress.csv')
df_exercise = pd.read_csv(data_dir / 'dataset_exercise.csv')

# %% FILTRAGEM
exclusoes_stress = ['f07', 'f13']
df_stress_clean = df_stress[~df_stress['subject_id'].isin(exclusoes_stress)].copy()
print(f"Conjunto de stress original: {len(df_stress)} \n-> Filtrado: {len(df_stress_clean)} ")

# %%
# Stress: 0=Repouso, 1=Stress
# Exercise: 2=Aeróbico, 3=Anaeróbico
df_ex_multi = df_exercise.copy()
df_ex_multi['label'] = df_ex_multi['label'].map({0: 2, 1: 3}) 
cols_comuns = df_stress_clean.columns.intersection(df_ex_multi.columns)
df_multi = pd.concat([df_stress_clean[cols_comuns], df_ex_multi[cols_comuns]], axis=0)


# %% TREINAMENTO
def rodar_experimento(nome_cenario, df, n_features_alvo, multiclasse=False):

    # 1. Separar X e y
    colunas_meta = ['subject_id', 'window_id', 'label', 'scenario', 'protocol']
    X = df.drop(columns=[c for c in colunas_meta if c in df.columns])
    y = df['label']

    # 2. Holdout 80-20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature Selection
    obj = 'multi:softmax' if multiclasse else 'binary:logistic'
    selector = RFE(estimator=XGBClassifier(random_state=42, n_jobs=-1), 
                   n_features_to_select=n_features_alvo, step=1)
    selector.fit(X_train_scaled, y_train)
    
    X_train_sel = selector.transform(X_train_scaled)
    X_test_sel = selector.transform(X_test_scaled)
    
    feat_names = X.columns[selector.support_]
    print(" Done!")
    print(f"   Top Features: {list(feat_names)}")
    
    # 5. Oversampling (SMOTE) - Apenas no Treino!
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train_sel, y_train)
    print(f" SMOTE (Train shape: {X_train_res.shape})")

    # Modelos
    models = {
        'KNN': KNeighborsClassifier(),
        'LogReg': LogisticRegression(max_iter=1000),
        'NaiveBayes': GaussianNB(),
        'SVM': SVC(probability=True),
        'RandomForest': RandomForestClassifier(random_state=42),
        'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }

    results = []
    
    # 7. Validação Cruzada (10-Fold)
    cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    
    scoring = {'acc': 'accuracy', 'f1': 'f1_weighted'}
    
    for name, model in models.items():
        # Cross Validation nos dados de treino (reamostrados)
        cv_results = cross_validate(model, X_train_res, y_train_res, cv=cv, scoring=scoring)
        
        # Teste Final (Holdout)
        model.fit(X_train_res, y_train_res)
        y_pred = model.predict(X_test_sel)
        
        acc_test = accuracy_score(y_test, y_pred)
        f1_test = f1_score(y_test, y_pred, average='weighted')
        
        results.append({
            'Model': name,
            'CV_Acc_Mean': cv_results['test_acc'].mean(),
            'CV_F1_Mean': cv_results['test_f1'].mean(),
            'Test_Acc': acc_test,
            'Test_F1': f1_test
        })
        
    return pd.DataFrame(results).sort_values(by='CV_Acc_Mean', ascending=False), feat_names

#%%
def comparar_balanceadores(titulo, df, random_state):
    X = df.drop(columns=['label', 'subject_id', 'window_id'], errors='ignore')
    X = X.select_dtypes(include=[np.number])
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state, stratify=y
    )
    
    # Pipeline: Imputar NaN -> Selecionar Features
    pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('selector', SelectKBest(f_classif, k=13))
    ])
    
    X_train_sel = pipeline.fit_transform(X_train, y_train)
    X_test_sel = pipeline.transform(X_test)
    
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

# %%
#  Stress vs Rest 13
res_stress, feat_stress = rodar_experimento("Classificação Stress/Repouso", df_stress_clean, 13)
display(res_stress)

#%%
#  Exercicio 12
res_ex, feat_ex = rodar_experimento("Classificação Aeróbico/Anaeróbico", df_exercise, 12)
display(res_ex)

#%%
# Multiclasse 19
res_multi, feat_multi = rodar_experimento("Classificação Multi-Classe", df_multi, 19, multiclasse=True)
display(res_multi)

# %% cOMPARATIVO
res_bal_stress = comparar_balanceadores("Stress - Impacto Balanceadores", df_stress_clean, 13)
display(res_bal_stress)

# %% 

sns.set_theme(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

def plot_custom_confusion_matrix(y_true, y_pred, classes, title, cmap):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    annot = np.empty_like(cm).astype(str)
    nrows, ncols = cm.shape
    for i in range(nrows):
        for j in range(ncols):
            c = cm[i, j]
            p = cm_norm[i, j]
            if i == j:
                s = cm.sum(axis=1)[i]
                annot[i, j] = '%.1f%%\n%d/%d' % (p * 100, c, s)
            elif c == 0:
                annot[i, j] = ''
            else:
                annot[i, j] = '%.1f%%\n%d' % (p * 100, c)
                
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(cm, annot=annot, fmt='', cmap=cmap, lw=0.5, ax=ax,
                xticklabels=classes, yticklabels=classes,
                annot_kws={"fontsize": 12, "fontweight": "bold"})
    
    ax.set_title(f"{title}", fontsize=14, fontweight='bold', pad=20)
    ax.set_xlabel('Predito', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    plt.tight_layout()
    plt.show()

def plot_feature_importance(model, feature_names, title, color):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1] # Decrescente

    df_imp = pd.DataFrame({
        'Feature': np.array(feature_names)[indices],
        'Importance': importances[indices]
    })

    fig, ax = plt.subplots(figsize=(10, len(feature_names) * 0.4 + 2))
    sns.barplot(data=df_imp, x='Importance', y='Feature', color=color, ax=ax)
    
    ax.set_title(f"Importância das Features: {title}", fontsize=14, fontweight='bold')
    ax.set_xlabel('Ganho (Importance)')
    ax.set_ylabel('')
    
    # Valores nas barras
    for i, v in enumerate(df_imp['Importance']):
        ax.text(v + 0.002, i, f"{v:.3f}", color='black', va='center', fontsize=9)

    plt.tight_layout()
    plt.show()

def visualizar_resultado_final(df, features_selecionadas, classes, title, cmap, color, balancer, multiclasse=False):

    
    # 1. Preparação (Replicando o pipeline com as features já selecionadas)
    colunas_meta = ['subject_id', 'window_id', 'label', 'scenario', 'protocol']
    X = df.drop(columns=[c for c in colunas_meta if c in df.columns])
    y = df['label']
    
    # Filtrar apenas as features que o RFE selecionou anteriormente
    X = X[features_selecionadas]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Balanceamento
    if balancer:
        print(f"Aplicando {balancer.__class__.__name__}...")
        X_train_res, y_train_res = balancer.fit_resample(X_train_scaled, y_train)
    else:
        X_train_res, y_train_res = X_train_scaled, y_train
        
    # Treino
    metric = 'mlogloss' if multiclasse else 'logloss'
    model = XGBClassifier(use_label_encoder=False, eval_metric=metric, random_state=42, n_estimators=200)
    model.fit(X_train_res, y_train_res)
    
    # Predição
    y_pred = model.predict(X_test_scaled)
    
    # Plots
    plot_custom_confusion_matrix(y_test, y_pred, classes, title, cmap)
    plot_feature_importance(model, features_selecionadas, title, color)

# %% 

# 1. Stress (SMOTE)
visualizar_resultado_final(
    df_stress_clean, 
    feat_stress, 
    classes=['Repouso', 'Stress'], 
    title="Stress vs Repouso", 
    cmap='Blues', 
    color='#1f77b4',
    balancer=SMOTE(random_state=42)
)

# 2. Exercise (SMOTE)
visualizar_resultado_final(
    df_exercise, 
    feat_ex, 
    classes=['Aeróbico', 'Anaeróbico'], 
    title="Aeróbico vs Anaeróbico", 
    cmap='Greens', 
    color='#2ca02c',
    balancer=SMOTE(random_state=42)
)

# 3. Multiclasse (SMOTE)
visualizar_resultado_final(
    df_multi, 
    feat_multi, 
    classes=['Repouso', 'Stress', 'Aeróbico', 'Anaeróbico'], 
    title="Classificação Multiclasse", 
    cmap='Oranges', 
    color='#ff7f0e',
    balancer=SMOTE(random_state=42),
    multiclasse=True
)



# %% SALVAR MODELOS TREINADOS

def salvar_melhor_modelo(df, features_selecionadas, nome_arquivo, balancer=None, multiclasse=False):
    """Treina e salva o melhor modelo (XGBoost) com joblib"""
    
    # Preparação
    colunas_meta = ['subject_id', 'window_id', 'label', 'scenario', 'protocol']
    X = df.drop(columns=[c for c in colunas_meta if c in df.columns])
    y = df['label']
    
    # Filtrar features selecionadas
    X = X[features_selecionadas]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Balanceamento
    if balancer:
        X_train_res, y_train_res = balancer.fit_resample(X_train_scaled, y_train)
    else:
        X_train_res, y_train_res = X_train_scaled, y_train
    
    # Treinar modelo
    metric = 'mlogloss' if multiclasse else 'logloss'
    model = XGBClassifier(
        use_label_encoder=False, 
        eval_metric=metric, 
        random_state=42, 
        n_estimators=200
    )
    model.fit(X_train_res, y_train_res)
    
    # Criar diretório se não existir
    models_dir = Path(r'C:\DEV\mestrado\WSD\experiments\models')
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Salvar modelo e scaler
    modelo_path = models_dir / f'{nome_arquivo}_model.joblib'
    scaler_path = models_dir / f'{nome_arquivo}_scaler.joblib'
    features_path = models_dir / f'{nome_arquivo}_features.joblib'
    
    joblib.dump(model, modelo_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(list(features_selecionadas), features_path)
    
    # Métricas no conjunto de teste
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"✅ Modelo salvo: {modelo_path}")
    print(f"   Acurácia: {acc:.4f} | F1-Score: {f1:.4f}")
    
    return model, scaler

#


# 1. Stress vs Repouso (SMOTEENN)
modelo_stress, scaler_stress = salvar_melhor_modelo(
    df_stress_clean, 
    feat_stress, 
    'stress_vs_rest',
    balancer=SMOTEENN(random_state=42)
)

modelo_exercise, scaler_exercise = salvar_melhor_modelo(
    df_exercise, 
    feat_ex, 
    'aerobic_vs_anaerobic',
    balancer=SMOTE(random_state=42)
)

modelo_multi, scaler_multi = salvar_melhor_modelo(
    df_multi, 
    feat_multi, 
    'multiclass',
    balancer=SMOTE(random_state=42),
    multiclasse=True
)


# %%
