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
df_stress = pd.read_csv(data_dir / 'dataset_stress.csv')
df_exercise = pd.read_csv(data_dir / 'dataset_exercise.csv')
#%%[markdown]
# Dados do Protocolo Stress
display(df_stress.head())
display(df_stress.columns)
display(df_stress.dtypes)
# %%
df_stress.drop(columns=['protocol'], inplace=True)
df_stress.info()
# %%
df_stress.describe()

# %%[markdown]
# Estatisticas Básicas
def calcular_estatisticas(df_stress):
    resultados = {}
    for coluna in df_stress.select_dtypes(include=['int64', 'float64']).columns:
        sequencia = df_stress[coluna].dropna()  # Remover valores ausentes

        # Tamanho do vetor
        m = len(sequencia)
        # Cálculos
        soma = np.sum(sequencia)
        multiplicacao = np.prod(sequencia)
        raiz_quadrada = np.sqrt(sequencia)
        potencia_2 = np.power(sequencia, 2)
        potencia_5 = np.power(sequencia, 5)
        maximo = np.max(sequencia)
        minimo = np.min(sequencia)
        dp = np.std(sequencia)
        dp_populacional = statistics.pstdev(sequencia)
        dp_amostral = statistics.stdev(sequencia)
        percentil_25 = np.percentile(sequencia, 25)
        percentil_50 = np.percentile(sequencia, 50)
        percentil_75 = np.percentile(sequencia, 75)
        quartis = statistics.quantiles(sequencia)
        frequencia = collections.Counter(sequencia)
        amplitude = maximo - minimo
        ponto_medio = (maximo + minimo) / 2
        moda = statistics.mode(sequencia) if len(set(sequencia)) < len(sequencia) else 'Nenhuma moda'
        media_ponderada = np.average(sequencia)
        mediana_1 = np.median(sequencia)
        mediana_2 = statistics.median(sequencia)
        media_aritmetica_1 = np.mean(sequencia)
        media_aritmetica_2 = statistics.mean(sequencia)
        media_quadratica = np.sqrt(np.sum(potencia_2) / m)

        # Calcular média geométrica apenas para dados positivos
        if (sequencia > 0).all():
            media_geometrica_1 = np.power(abs(multiplicacao), (1 / m))
            media_geometrica_2 = statistics.geometric_mean(sequencia)
        else:
            media_geometrica_1 = 'N/A'
            media_geometrica_2 = 'N/A'

        #media_harmonica = statistics.harmonic_mean(sequencia) if len(sequencia) > 0 else 'N/A'
        di = percentil_75 - percentil_25
        limite_inferior = percentil_25 - (1.5 * di)
        limite_superior = percentil_75 + (1.5 * di)  # Corrigido para ser superior
        variancia_amostral = statistics.variance(sequencia)
        variancia_populacional = statistics.pvariance(sequencia)
        variancia_1 = np.var(sequencia)
        cv = dp / media_aritmetica_1 if media_aritmetica_1 != 0 else 'N/A'

        # Armazenar resultados
        resultados[coluna] = {
            'Tamanho': m,
            'Soma': soma,
            'Multiplicação': multiplicacao,
            'Raiz Quadrada': raiz_quadrada.tolist(),
            'Potência 2': potencia_2.tolist(),
            'Potência 5': potencia_5.tolist(),
            'Máximo': maximo,
            'Mínimo': minimo,
            'Desvio Padrão': dp,
            'Desvio Padrão Populacional': dp_populacional,
            'Desvio Padrão Amostral': dp_amostral,
            'Percentil 25': percentil_25,
            'Percentil 50': percentil_50,
            'Percentil 75': percentil_75,
            'Quartis': quartis,
            'Frequência': dict(frequencia),
            'Amplitude': amplitude,
            'Ponto Médio': ponto_medio,
            'Moda': moda,
            'Média Ponderada': media_ponderada,
            'Mediana (NumPy)': mediana_1,
            'Mediana (Statistics)': mediana_2,
            'Média Aritmética (NumPy)': media_aritmetica_1,
            'Média Aritmética (Statistics)': media_aritmetica_2,
            'Média Quadrática': media_quadratica,
            'Média Geométrica (NumPy)': media_geometrica_1,
            'Média Geométrica (Statistics)': media_geometrica_2,
            #'Média Harmônica': media_harmonica,
            'Diferença Interquartil': di,
            'Limite Inferior': limite_inferior,
            'Limite Superior': limite_superior,
            'Variância Amostral': variancia_amostral,
            'Variância Populacional': variancia_populacional,
            'Variância (NumPy)': variancia_1,
            'Coeficiente de Variação': cv
        }

    return resultados

estatisticas = calcular_estatisticas(df_stress)
for coluna, stats in estatisticas.items():
    print(f"Coluna: {coluna}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()
# %% [markdown]
# Contagem de valores Unicos
df_stress.nunique()
# %%
for col in df_stress.columns:
    print(f'\nColuna: {col}')
    print(f'Valores únicos ({df_stress[col].nunique()}): {df_stress[col].unique()}')

#%%
for col in df_stress.columns:
    print(f"\nColuna: {col}")
    print(df_stress[col].value_counts())
# %%
numeric_columns = df_stress.select_dtypes(include='number').columns
categorical_columns = df_stress.select_dtypes(include=['object', 'category']).columns
print(numeric_columns)
print(categorical_columns)

# %% [markdown]
# Analise de Atributos Numéricos

def plot_histograms_with_kde(df_stress, num_cols=2, num_rows=None):
    
    sns.set_theme()
    numeric_columns = df_stress.select_dtypes(include=np.number).columns
    num_plots = len(numeric_columns)

    if num_rows is None:
        num_rows = (num_plots // num_cols) + (num_plots % num_cols > 0)

    figsize = (num_cols * 5, num_rows * 4)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, column in enumerate(numeric_columns):
        if df_stress[column].var() == 0:
            axes[i].text(0.5, 0.5, f'Variância zero\n({column})',
                         horizontalalignment='center',
                         verticalalignment='center')
            axes[i].set_title(f"Histograma: {column}")
            axes[i].axis('off')
            continue

        sns.histplot(data=df_stress, x=column, kde=True, ax=axes[i])
        axes[i].set_title("Histograma: " + column)
        kde = sns.kdeplot(data=df_stress[column], ax=axes[i], color='red', lw=2)
        kde_data = kde.lines[0].get_data()

        if len(kde_data[0]) > 0:
            kde_peak = kde_data[0][kde_data[1].argmax()]
            axes[i].axvline(kde_peak, color='green', linestyle='--', lw=2)
            axes[i].text(kde_peak, 0.9 * max(kde_data[1]), f'Pico: {kde_peak:.2f}', color='green')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

plot_histograms_with_kde(df_stress, num_cols=5, num_rows=11)
# %%
def plotar_grid_boxplots(df, colunas_ignorar=None, n_colunas_grid=3):

    sns.set_theme(style="whitegrid")
    total_plots = len(numeric_columns)
    n_linhas = math.ceil(total_plots / n_colunas_grid)
    fig, axes = plt.subplots(n_linhas, n_colunas_grid, 
                             figsize=(6 * n_colunas_grid, 5 * n_linhas))
    axes = axes.flatten() 
    boxplot_color = '#836FFF'

    for i, coluna in enumerate(numeric_columns):
        ax = axes[i]
        sns.boxplot(data=df, y=coluna, ax=ax, color=boxplot_color,
                    flierprops=dict(marker='o', color='red', markersize=3, alpha=0.5))
        mean = df[coluna].mean()
        std_dev = df[coluna].std()
        ax.scatter([0], [mean], color='black', zorder=10, s=80, marker='o')
        ax.errorbar(x=[0], y=[mean], yerr=[std_dev], fmt='none', ecolor='orange',
                    elinewidth=3, capsize=8, zorder=11)
        ax.set_title(coluna, fontweight='bold', fontsize=11)
        ax.set_ylabel('')
        ax.set_xlabel('')
        
        stats_text = f"Média: {mean:.2f}\nDesvio: {std_dev:.2f}"
        ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, 
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8), fontsize=9)

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

#%%
plotar_grid_boxplots(df_stress, n_colunas_grid=3)
# %% QQ Plot

def qq_plot(df_stress, ax, distribution='norm', xlabel='Teóricos Quantiles',
            ylabel='Quantiles Observados'):

    stats.probplot(df_stress, dist=distribution, plot=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_qq_for_numeric_columns(df_stress, nrows=1, ncols=1):

    numeric_columns = df_stress.select_dtypes(include=[np.number]).columns

    if nrows * ncols < len(numeric_columns):
        raise ValueError("\nO número total de subplots deve ser maior ou igual ao número de colunas numéricas.")


    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows)) 

    if nrows * ncols == 1:
        axes = np.array([[axes]])

    for ax, column in zip(axes.flatten(), numeric_columns):
        qq_plot(df_stress[column], ax, xlabel='Teóricos Quantiles', ylabel='Quantiles Observados')
        ax.set_title(f'QQ Plot - {column}')

    plt.tight_layout() 
    plt.show()

#%%
from scipy import stats
plot_qq_for_numeric_columns(df_stress, 17, 3)


# %% Testar a Normalidade

def testar_normalidade_completa(df_stress):
    numeric_cols = df_stress.select_dtypes(include='number').columns

    for col in numeric_cols:
        dados = df_stress[col].dropna()
        print(f'\n=== Coluna: {col} ===')

        # Shapiro-Wilk
        stat, p = shapiro(dados)
        print(f'Shapiro-Wilk: Stat={stat:.3f}, p={p:.3f} --> Normal? {"Sim" if p > 0.05 else "Não"}')

        # D'Agostino's K^2
        stat, p = normaltest(dados)
        print(f"D'Agostino's K²: Stat={stat:.3f}, p={p:.3f} --> Normal? {'Sim' if p > 0.05 else 'Não'}")

        # Kolmogorov-Smirnov
        # Ajustando média e desvio para a normal comparada
        mean = np.mean(dados)
        std = np.std(dados, ddof=1)
        stat, p = kstest(dados, 'norm', args=(mean, std))
        print(f'Kolmogorov-Smirnov: Stat={stat:.3f}, p={p:.3f} --> Normal? {"Sim" if p > 0.05 else "Não"}')

        # Anderson-Darling
        result = anderson(dados)
        print(f'Anderson-Darling: Stat={result.statistic:.3f}')
        for sl, cv in zip(result.significance_level, result.critical_values):
            status = 'Rejeita H0' if result.statistic > cv else 'Não rejeita H0'
            print(f'  Nível {sl}% - Valor crítico: {cv:.3f} --> {status}')
        
        # Mann-Whitney U (Não Parametrico)
        if 'label' in df_stress.columns and col != 'label':
            grupo0 = df_stress[df_stress['label'] == 0][col]
            grupo1 = df_stress[df_stress['label'] == 1][col]
            
            if len(grupo0) > 0 and len(grupo1) > 0:
                stat, p = mannwhitneyu(grupo0, grupo1)
                print(f'Mann-Whitney U: Stat={stat:.3f}, p={p:.3f} --> Diferença Significativa? {"Sim" if p < 0.05 else "Não"}')
        
        if result.statistic > result.critical_values[2]: 
            print('Conclusão geral Anderson-Darling: Provavelmente NÃO normal')
        else:
            print('Conclusão geral Anderson-Darling: Provavelmente normal')
        
        

testar_normalidade_completa(df_stress)

# %% Análise de Correlação entre os Atributos Numéricos

num_df = df_stress[numeric_columns]
correlation = num_df.corr()
correlation
# %%
plt.figure(figsize=(30, 24))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.2)
plt.title('Matriz de Correlação entre atributos numéricos')
plt.show()

# %%

top_n = 10
correlation.head(top_n)
# %% Analise de Colinearidade 

X = df_stress[numeric_columns].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

vif_data = pd.DataFrame()
vif_data["feature"] = numeric_columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
vif_data

# %% Atributos CAtegóricos

def plot_categorical_distribution(df_stress, num_rows=2, num_cols=2, rotation=45):
    """
    Plota a distribuição de colunas categóricas em um DataFrame utilizando gráficos de barras,
    exibindo os valores das contagens sobre as barras. Permite configurar o número de linhas e
    colunas da grade de subplots (aceita qualquer valor ≥1) e o ângulo de rotação dos rótulos.

    Parâmetros:
    -----------
    df_stress : pandas.DataFrame
        DataFrame de entrada contendo os dados.

    num_rows : int, opcional (padrão=2)
        Número de linhas na grade de subplots (mínimo 1).

    num_cols : int, opcional (padrão=2)
        Número de colunas na grade de subplots (mínimo 1).

    rotation : int, opcional (padrão=45)
        Ângulo de rotação dos rótulos das categorias no eixo x.

    Retorno:
    --------
    None
        A função apenas exibe os gráficos e não retorna valores.

    Exemplo de uso:
    ---------------
    plot_categorical_distribution(adult_dataset, num_rows=1, num_cols=3, rotation=90)
    """

    # Configura tema padrão do seaborn
    sns.set_theme()

    # Seleciona colunas categóricas do DataFrame
    categorical_columns = df_stress.select_dtypes(include='object').columns

    # Verifica se existem colunas categóricas
    if len(categorical_columns) == 0:
        print("Nenhuma coluna categórica encontrada para plotar.")
        return

    # Calcula o número total de gráficos disponíveis no grid
    total_plots = num_rows * num_cols

    # Define quantas colunas serão efetivamente plotadas
    num_to_plot = min(len(categorical_columns), total_plots)

    # Ajusta dinamicamente o tamanho da figura (largura e altura)
    plt.rcParams['figure.figsize'] = [num_cols * 6, num_rows * 5]

    # Cria a grade de subplots
    fig, axes = plt.subplots(num_rows, num_cols)
    # Se só houver um subplot (caso 1x1), encapsula em lista
    if total_plots == 1:
        axes = [axes]
    else:
        axes = axes.flatten()  # Transforma em lista linear para iterar

    # Itera sobre as colunas categóricas a serem plotadas
    for i in range(num_to_plot):
        column = categorical_columns[i]

        # Conta as ocorrências de cada categoria
        contagem = df_stress[column].value_counts()

        # Prepara DataFrame auxiliar com as contagens
        contagem_df_stress = pd.DataFrame({'Categoria': contagem.index, 'Frequência': contagem.values})

        # Plota gráfico de barras com cores por categoria via hue
        sns.barplot(data=contagem_df_stress, x='Categoria', y='Frequência',
                    hue='Categoria', ax=axes[i], palette='viridis', legend=False)

        # Define título e rótulos dos eixos
        axes[i].set_title("Distribuição: " + column, fontsize=14)
        axes[i].set_xlabel('Categoria', fontsize=12)
        axes[i].set_ylabel('Frequência', fontsize=12)

        # Rotaciona os rótulos do eixo x
        plt.setp(axes[i].get_xticklabels(), rotation=rotation, ha='right')

        # Exibe os valores sobre as barras
        for idx, value in enumerate(contagem_df_stress['Frequência']):
            axes[i].text(idx, value, f'{value}', ha='center', va='bottom', fontsize=11, color='black')

    # Remove subplots sobrando, caso existam mais slots do que colunas categóricas
    for j in range(num_to_plot, total_plots):
        fig.delaxes(axes[j])

    # Ajusta o layout para evitar sobreposição
    plt.tight_layout()

    # Exibe os gráficos
    plt.show()

plot_categorical_distribution(df_stress, num_rows=4, num_cols=2)
# %%
for col in categorical_columns:
    contingency_table = pd.crosstab(df_stress[col], df_stress['subject_id'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"Variável: {col}, p-valor: {p:.5f} -- ", "Associação significativa" if p < 0.05 else "Sem associação significativa")

# %% Analise de Outliers
def analisar_outliers(df_stress, tamanho_figura=(10, 10), num_cols=3):
    """
    Analisa outliers em um DataFrame, removendo a primeira coluna e criando boxplots para colunas numéricas.

    Parameters:
    df_stress (pd.DataFrame): DataFrame a ser analisado.
    tamanho_figura (tuple): Tamanho da figura para os subplots.
    num_cols (int): Número de colunas para os subplots.
    """
    # Remove a primeira coluna
    df_stress = df_stress.iloc[:, 1:]

    # Identifica colunas numéricas
    numeric_columns = df_stress.select_dtypes(include='number').columns

    # Configuração para o tamanho da figura e tema
    plt.rcParams['figure.figsize'] = tamanho_figura
    sns.set_theme()

    # Define o número de colunas e linhas para os subplots
    num_plots = len(numeric_columns)
    num_rows = (num_plots + num_cols - 1) // num_cols

    # Cria uma grade de subplots com o número calculado de linhas e colunas
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(num_cols * 7, num_rows * 5))
    axes = axes.flatten()

    # Define a cor do boxplot
    boxplot_color = '#836FFF'

    # Análise de Outliers
    for i, column in enumerate(numeric_columns):
        ax = axes[i]

        # Boxplot
        sns.boxplot(data=df_stress, y=column, ax=ax, color=boxplot_color,
                    flierprops=dict(marker='o', color='red', markersize=5))
        ax.set_title(f'Boxplot: {column}')
        ax.set_ylabel('Valor')

        # Identificar outliers usando IQR
        Q1 = df_stress[column].quantile(0.25)
        Q3 = df_stress[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Adicionar linhas para os limites
        ax.axhline(lower_bound, color='red', linestyle='--', label='Limite Inferior')
        ax.axhline(upper_bound, color='red', linestyle='--', label='Limite Superior')

        # Adicionar a legenda fora do gráfico
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

        # Outliers usando Z-Score
        mean = df_stress[column].mean()
        std_dev = df_stress[column].std()
        if std_dev > 0:  # Verifica se o desvio padrão não é zero
            z_scores = (df_stress[column] - mean) / std_dev
            outliers_z_score = df_stress[column][np.abs(z_scores) > 3]  # Z-score > 3
        else:
            outliers_z_score = pd.Series(dtype='float64')  # Se o desvio padrão for zero, não há outliers

        # Adicionar os outliers do Z-Score ao gráfico
        if not outliers_z_score.empty:
            ax.scatter(outliers_z_score.index, outliers_z_score, color='orange', s=100, label='Outliers (Z-Score)')

        # Outliers usando IQR
        outliers_iqr = df_stress[column][(df_stress[column] < lower_bound) | (df_stress[column] > upper_bound)]

        # Adicionar os outliers do IQR ao gráfico
        if not outliers_iqr.empty:
            ax.scatter(outliers_iqr.index, outliers_iqr, color='green', s=100, label='Outliers (IQR)')

    # Remove subplots não utilizados
    for ax in axes[num_plots:]:
        fig.delaxes(ax)

    # Ajusta o layout para não sobrepor os gráficos
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.show()

    # Exibe os outliers encontrados
    print("\n\n=====================================================================")
    print("\nOutliers identificados usando Z-Score:")
    for column in numeric_columns:
        mean = df_stress[column].mean()
        std_dev = df_stress[column].std()
        if std_dev > 0:  # Verifica se o desvio padrão não é zero
            z_scores = (df_stress[column] - mean) / std_dev
            outliers_z_score = df_stress[column][np.abs(z_scores) > 3]
            if not outliers_z_score.empty:
                print(f"\n{column}: {outliers_z_score}")
            else:
                print(f"\n{GREEN}\033[1m{column}: Nenhum outlier encontrado usando Z-Score{RESET}")
        else:
            print(f"\n{column}: Dados insuficientes para cálculo de Z-Score")

    print("\n\n=====================================================================")
    print("\nOutliers identificados usando IQR:")
    for column in numeric_columns:
        Q1 = df_stress[column].quantile(0.25)
        Q3 = df_stress[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_iqr = df_stress[column][(df_stress[column] < lower_bound) | (df_stress[column] > upper_bound)]
        if not outliers_iqr.empty:
            print(f"\n{column}: {outliers_iqr}")
        else:
            print(f"\n{GREEN}\033[1m{column}: Nenhum outlier encontrado usando IQR{RESET}")

    print("\n\n=====================================================================")

#%%
analisar_outliers(df_stress, tamanho_figura=(10, 10), num_cols=3)

# %%
