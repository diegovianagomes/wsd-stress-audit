#%% Importações
import sys
from pathlib import Path
src_path = Path(__file__).resolve().parent.parent.parent / 'src'
sys.path.insert(0, str(src_path))
from config_imports import *

# Cores ANSI
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[0m"

#%%[markdown]
# Carrega o conjunto de dados
current_dir = Path(os.getcwd())
processed_dir = current_dir / 'experiments' / 'processed'
print(processed_dir)

data_dir = Path.cwd().parent / 'processed'
df_exercise = pd.read_csv(data_dir / 'dataset_exercise.csv')
#%%[markdown]
# Conjunto de dados brutos do Protocolo Stress
display(df_exercise.head())
display(df_exercise.columns)
display(df_exercise.dtypes)
# %% [markdown]
# Suprime a coluna 'protocol' que é apenas util nos protocolos Exercicio
df_exercise.drop(columns=['protocol'], inplace=True)
df_exercise.info()
# %%
df_exercise.describe()
#%%

# %%[markdown]
# Estatisticas Básicas
def calcular_estatisticas(df_exercise):
    resultados = {}
    for coluna in df_exercise.select_dtypes(include=['int64', 'float64']).columns:
        sequencia = df_exercise[coluna].dropna()  # Remover valores ausentes

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
        limite_superior = percentil_75 + (1.5 * di)
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

estatisticas = calcular_estatisticas(df_exercise)
for coluna, stats in estatisticas.items():
    print(f"Coluna: {coluna}")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()
# %% [markdown]
# Contagem de valores Unicos
df_exercise.nunique()
# %%
for col in df_exercise.columns:
    print(f'\nColuna: {col}')
    print(f'Valores únicos ({df_exercise[col].nunique()}): {df_exercise[col].unique()}')

#%%
for col in df_exercise.columns:
    print(f"\nColuna: {col}")
    print(df_exercise[col].value_counts())
# %%
numeric_columns = df_exercise.select_dtypes(include='number').columns
categorical_columns = df_exercise.select_dtypes(include=['object', 'category']).columns
print(numeric_columns)
print(categorical_columns)

# %% [markdown]
# Analise de Atributos Numéricos

def plot_histograms_with_kde(df_exercise, num_cols=2, num_rows=None):
    
    sns.set_theme()
    numeric_columns = df_exercise.select_dtypes(include=np.number).columns
    num_plots = len(numeric_columns)

    if num_rows is None:
        num_rows = (num_plots // num_cols) + (num_plots % num_cols > 0)

    figsize = (num_cols * 5, num_rows * 4)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=figsize)
    axes = axes.flatten()

    for i, column in enumerate(numeric_columns):
        if df_exercise[column].var() == 0:
            axes[i].text(0.5, 0.5, f'Variância zero\n({column})',
                         horizontalalignment='center',
                         verticalalignment='center')
            axes[i].set_title(f"Histograma: {column}")
            axes[i].axis('off')
            continue

        sns.histplot(data=df_exercise, x=column, kde=True, ax=axes[i])
        axes[i].set_title("Histograma: " + column)
        kde = sns.kdeplot(data=df_exercise[column], ax=axes[i], color='red', lw=2)
        kde_data = kde.lines[0].get_data()

        if len(kde_data[0]) > 0:
            kde_peak = kde_data[0][kde_data[1].argmax()]
            axes[i].axvline(kde_peak, color='green', linestyle='--', lw=2)
            axes[i].text(kde_peak, 0.9 * max(kde_data[1]), f'Pico: {kde_peak:.2f}', color='green')

    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()

plot_histograms_with_kde(df_exercise, num_cols=3, num_rows=17)


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

#%%[markdown]
# QQ Plot
plotar_grid_boxplots(df_exercise, n_colunas_grid=3)
# %% QQ Plot

def qq_plot(df_exercise, ax, distribution='norm', xlabel='Teóricos Quantiles',
            ylabel='Quantiles Observados'):

    stats.probplot(df_exercise, dist=distribution, plot=ax)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

def plot_qq_for_numeric_columns(df_exercise, nrows=1, ncols=1):

    numeric_columns = df_exercise.select_dtypes(include=[np.number]).columns

    if nrows * ncols < len(numeric_columns):
        raise ValueError("\nO número total de subplots deve ser maior ou igual ao número de colunas numéricas.")


    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows)) 

    if nrows * ncols == 1:
        axes = np.array([[axes]])

    for ax, column in zip(axes.flatten(), numeric_columns):
        qq_plot(df_exercise[column], ax, xlabel='Teóricos Quantiles', ylabel='Quantiles Observados')
        ax.set_title(f'QQ Plot - {column}')

    plt.tight_layout() 
    plt.show()

#%%
from scipy import stats
plot_qq_for_numeric_columns(df_exercise, 17, 3)


# %% [markdown]
# Testar a Normalidade, foi adicionado Mann-Whitney U para testes não parametricos

def testar_normalidade_completa(df_exercise):
    numeric_cols = df_exercise.select_dtypes(include='number').columns

    for col in numeric_cols:
        dados = df_exercise[col].dropna()
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
        if 'label' in df_exercise.columns and col != 'label':
            grupo0 = df_exercise[df_exercise['label'] == 0][col]
            grupo1 = df_exercise[df_exercise['label'] == 1][col]
            
            if len(grupo0) > 0 and len(grupo1) > 0:
                stat, p = mannwhitneyu(grupo0, grupo1)
                print(f'Mann-Whitney U: Stat={stat:.3f}, p={p:.3f} --> Diferença Significativa? {"Sim" if p < 0.05 else "Não"}')
        
        if result.statistic > result.critical_values[2]: 
            print('Conclusão geral Anderson-Darling: Provavelmente NÃO normal')
        else:
            print('Conclusão geral Anderson-Darling: Provavelmente normal')
        
        

testar_normalidade_completa(df_exercise)

# %% [markdown]
# Análise de Correlação entre os Atributos Numéricos

num_df = df_exercise[numeric_columns]
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
# %% [markdown]
# Analise de Colinearidade 

X = df_exercise[numeric_columns].copy()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

vif_data = pd.DataFrame()
vif_data["feature"] = numeric_columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
vif_data

# %% [markdown]
# Atributos CAtegóricos

def plot_categorical_distribution(df_exercise, num_rows=2, num_cols=2, rotation=45):
    """
    Plota a distribuição de colunas categóricas em um DataFrame utilizando gráficos de barras,
    exibindo os valores das contagens sobre as barras. Permite configurar o número de linhas e
    colunas da grade de subplots (aceita qualquer valor ≥1) e o ângulo de rotação dos rótulos.

    Parâmetros:
    -----------
    df_exercise : pandas.DataFrame
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
    categorical_columns = df_exercise.select_dtypes(include='object').columns

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
        contagem = df_exercise[column].value_counts()

        # Prepara DataFrame auxiliar com as contagens
        contagem_df_exercise = pd.DataFrame({'Categoria': contagem.index, 'Frequência': contagem.values})

        # Plota gráfico de barras com cores por categoria via hue
        sns.barplot(data=contagem_df_exercise, x='Categoria', y='Frequência',
                    hue='Categoria', ax=axes[i], palette='viridis', legend=False)

        # Define título e rótulos dos eixos
        axes[i].set_title("Distribuição: " + column, fontsize=14)
        axes[i].set_xlabel('Categoria', fontsize=12)
        axes[i].set_ylabel('Frequência', fontsize=12)

        # Rotaciona os rótulos do eixo x
        plt.setp(axes[i].get_xticklabels(), rotation=rotation, ha='right')

        # Exibe os valores sobre as barras
        for idx, value in enumerate(contagem_df_exercise['Frequência']):
            axes[i].text(idx, value, f'{value}', ha='center', va='bottom', fontsize=11, color='black')

    # Remove subplots sobrando, caso existam mais slots do que colunas categóricas
    for j in range(num_to_plot, total_plots):
        fig.delaxes(axes[j])

    # Ajusta o layout para evitar sobreposição
    plt.tight_layout()

    # Exibe os gráficos
    plt.show()

plot_categorical_distribution(df_exercise, num_rows=4, num_cols=2)
# %%
for col in categorical_columns:
    contingency_table = pd.crosstab(df_exercise[col], df_exercise['subject_id'])
    chi2, p, _, _ = chi2_contingency(contingency_table)
    print(f"Variável: {col}, p-valor: {p:.5f} -- ", "Associação significativa" if p < 0.05 else "Sem associação significativa")

# %% [markdown]
# Analise de Outliers
def analisar_outliers(df_exercise, tamanho_figura=(10, 10), num_cols=3):
    """
    Analisa outliers em um DataFrame, removendo a primeira coluna e criando boxplots para colunas numéricas.

    Parameters:
    df_exercise (pd.DataFrame): DataFrame a ser analisado.
    tamanho_figura (tuple): Tamanho da figura para os subplots.
    num_cols (int): Número de colunas para os subplots.
    """
    # Remove a primeira coluna
    df_exercise = df_exercise.iloc[:, 1:]

    # Identifica colunas numéricas
    numeric_columns = df_exercise.select_dtypes(include='number').columns

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
        sns.boxplot(data=df_exercise, y=column, ax=ax, color=boxplot_color,
                    flierprops=dict(marker='o', color='red', markersize=5))
        ax.set_title(f'Boxplot: {column}')
        ax.set_ylabel('Valor')

        # Identificar outliers usando IQR
        Q1 = df_exercise[column].quantile(0.25)
        Q3 = df_exercise[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Adicionar linhas para os limites
        ax.axhline(lower_bound, color='red', linestyle='--', label='Limite Inferior')
        ax.axhline(upper_bound, color='red', linestyle='--', label='Limite Superior')

        # Adicionar a legenda fora do gráfico
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1), frameon=False)

        # Outliers usando Z-Score
        mean = df_exercise[column].mean()
        std_dev = df_exercise[column].std()
        if std_dev > 0:  # Verifica se o desvio padrão não é zero
            z_scores = (df_exercise[column] - mean) / std_dev
            outliers_z_score = df_exercise[column][np.abs(z_scores) > 3]  # Z-score > 3
        else:
            outliers_z_score = pd.Series(dtype='float64')  # Se o desvio padrão for zero, não há outliers

        # Adicionar os outliers do Z-Score ao gráfico
        if not outliers_z_score.empty:
            ax.scatter(outliers_z_score.index, outliers_z_score, color='orange', s=100, label='Outliers (Z-Score)')

        # Outliers usando IQR
        outliers_iqr = df_exercise[column][(df_exercise[column] < lower_bound) | (df_exercise[column] > upper_bound)]

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
        mean = df_exercise[column].mean()
        std_dev = df_exercise[column].std()
        if std_dev > 0:  # Verifica se o desvio padrão não é zero
            z_scores = (df_exercise[column] - mean) / std_dev
            outliers_z_score = df_exercise[column][np.abs(z_scores) > 3]
            if not outliers_z_score.empty:
                print(f"\n{column}: {outliers_z_score}")
            else:
                print(f"\n{GREEN}\033[1m{column}: Nenhum outlier encontrado usando Z-Score{RESET}")
        else:
            print(f"\n{column}: Dados insuficientes para cálculo de Z-Score")

    print("\n\n=====================================================================")
    print("\nOutliers identificados usando IQR:")
    for column in numeric_columns:
        Q1 = df_exercise[column].quantile(0.25)
        Q3 = df_exercise[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers_iqr = df_exercise[column][(df_exercise[column] < lower_bound) | (df_exercise[column] > upper_bound)]
        if not outliers_iqr.empty:
            print(f"\n{column}: {outliers_iqr}")
        else:
            print(f"\n{GREEN}\033[1m{column}: Nenhum outlier encontrado usando IQR{RESET}")

    print("\n\n=====================================================================")

#%%
analisar_outliers(df_exercise, tamanho_figura=(10, 10), num_cols=3)

# %%
#%%[markdown]
# Roda o OLS

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

X = df_exercise[features_significativas]
X = sm.add_constant(X)

y = df_exercise['label']

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


# %% [markdown]
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
X = df_exercise.drop(columns=['label'])
X
#%%
y = df_exercise['label']
y
# %%
kf = KFold(n_splits=10, shuffle=True, random_state=42)
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
y = le.fit_transform(df_exercise['label'])
print(f"\nMapeamento: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Remover colunas não-numéricas de X
X = df_exercise.drop(columns=['label', 'subject_id', 'window_id'], errors='ignore')
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

y = pd.Series(y, index=df_exercise.index, name='label')
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

#%%