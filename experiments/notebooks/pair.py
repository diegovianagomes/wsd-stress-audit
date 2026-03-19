#%%

import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


BASE_DIR = Path(__file__).resolve().parent.parent
RF_RESULTS = os.path.join(BASE_DIR, "folds", "grid_search_rf", "Evaluation_results.csv")
XGB_RESULTS = os.path.join(BASE_DIR, "folds", "grid_search_xgb", "xgb_final_evaluation.csv")

#%%
df_rf  = pd.read_csv(RF_RESULTS)
df_xgb = pd.read_csv(XGB_RESULTS)


# %%
df_rf.columns = [c.lower().strip() for c in df_rf.columns]
df_xgb.columns = [c.lower().strip() for c in df_xgb.columns]
#%%
df_rf.columns

# %%
required_cols = ['fold', 'f1', 'roc_auc', 'precision', 'recall']

for df, name in [(df_rf, 'RF'), (df_xgb, 'XGB')]:
    print(f'Colunas: {df.columns.tolist()}')

df_compare = (
    df_rf[required_cols]
    .merge(
        df_xgb[required_cols],
        on='fold',
        suffixes=('_rf', '_xgb')
    )
)
display(df_compare.head(10))

# %%

metricas = pd.DataFrame({
    'Modelo': ['Random Forest', 'XGBoost'],
})

stats_metricas = {
    'roc_auc': 'ROC AUC',
    'f1': 'F1',
    'precision': 'Precision',
    'recall': 'Recall',
}

for metric, label in stats_metricas.items():
    metricas[f'Média {label}'] = [
        df_compare[f'{metric}_rf'].mean(),
        df_compare[f'{metric}_xgb'].mean(),
    ]
    metricas[f'Desvio Padrão {label}'] = [
        df_compare[f'{metric}_rf'].std(),
        df_compare[f'{metric}_xgb'].std(),
    ]
    metricas[f'Min {label}'] = [
        df_compare[f'{metric}_rf'].min(),
        df_compare[f'{metric}_xgb'].min(),
    ]
    metricas[f'Max {label}'] = [
        df_compare[f'{metric}_rf'].max(),
        df_compare[f'{metric}_xgb'].max(),
    ]
metricas

# %% [markdown]
# Diferencia entre os modelos (XGBoost - Random Forest)

diff_stats = []
for metric, label in stats_metricas.items():
    df_compare[f'diff_{metric}'] = df_compare[f'{metric}_xgb'] - df_compare[f'{metric}_rf']
    diff_stats.append({
        'Métrica': label,
        'Diff media (XGB - RF)': df_compare[f'diff_{metric}'].mean(),
        'Desvio Padrão': df_compare[f'diff_{metric}'].std(),
    })

df_diff_stats = pd.DataFrame(diff_stats)
df_diff_stats

# %% [markdown]
# Teste de Hipótese para cada métrica

ttest_results = []
for metric, label in stats_metricas.items():
    diff = df_compare[f'{metric}_rf'] - df_compare[f'{metric}_xgb']
    _, p_shapiro = stats.shapiro(diff)
    normal = p_shapiro >= 0.05

    if normal:
        t_stat, p_value = stats.ttest_rel(
            df_compare[f'{metric}_rf'],
            df_compare[f'{metric}_xgb']
        )
        test_used = 'paired t-test'
    else:
        t_stat, p_value = stats.wilcoxon(
            df_compare[f'{metric}_rf'],
            df_compare[f'{metric}_xgb'],
            zero_method='wilcox'
        )
        test_used = 'Wilcoxon'

    ttest_results.append({
        'Métrica': label,
        'Shapiro p-value': round(p_shapiro, 4),
        'Normalidade': 'Sim' if normal else 'Não',
        'Teste Usado': test_used,
        't_stat (RF - XGB)': t_stat,
        'p-value': p_value,
        'Significativo (p < 0.05)': 'Sim' if p_value < 0.05 else 'Não'
    })

df_ttest = pd.DataFrame(ttest_results)
df_ttest

# %% [markdown]
# Teste Não-Paramétrico Wilcoxon Signed-Rank

metrics = [
    ('ROC AUC', 'roc_auc_rf', 'roc_auc_xgb'),
    ('F1 Score', 'f1_rf', 'f1_xgb'),
    ('Precision', 'precision_rf', 'precision_xgb'),
    ('Recall', 'recall_rf', 'recall_xgb')
]

wilcoxon_results = []
for name, col_rf, col_xgb in metrics:
    stat, p_val = stats.wilcoxon(df_compare[col_rf], df_compare[col_xgb], zero_method='wilcox')
    wilcoxon_results.append({
        'Métrica': name,
        'w_stat': stat,
        'p-value': p_val,
        'Significativo (p < 0.05)': 'Sim' if p_val < 0.05 else 'Não'
    })

df_wilcoxon = pd.DataFrame(wilcoxon_results)
df_wilcoxon


# %% [markdown]
# Resultados

plt.style.use('seaborn-v0_8-whitegrid')
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

colors = ['#1f77b4', '#ff7f0e']

#
for idx, (metric, label) in enumerate(stats_metricas.items()):
    data_to_plot = [df_compare[f'{metric}_rf'], df_compare[f'{metric}_xgb']]
    bp = axes[idx].boxplot(data_to_plot, labels=['Random Forest', 'XGBoost'], 
                            patch_artist=True, widths=0.6)
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    
    axes[idx].set_title(f'Distribuição de {label}', 
                        fontsize=12, 
                        fontweight='bold'
    )
    axes[idx].set_ylabel(label, fontsize=10)
    axes[idx].grid(axis='y', linestyle='--', alpha=0.7)
    means = [df_compare[f'{metric}_rf'].mean(), df_compare[f'{metric}_xgb'].mean()]
    axes[idx].plot([1, 2], means, 'o', color='red', markersize=8, label='Média')
    axes[idx].legend()

plt.tight_layout()
plt.show()
# %%
# Performance em cada Fold para todas as métricas
fig2, axes2 = plt.subplots(2, 2, figsize=(14, 10))
axes2 = axes2.flatten()

colors_rf_xgb = ['#1f77b4', '#ff7f0e']

for idx, (metric, label) in enumerate(stats_metricas.items()):
    axes2[idx].plot(df_compare['fold'], df_compare[f'{metric}_rf'], 
                    marker='o', label='Random Forest', color=colors_rf_xgb[0], linewidth=2)
    axes2[idx].plot(df_compare['fold'], df_compare[f'{metric}_xgb'], 
                    marker='s', label='XGBoost', color=colors_rf_xgb[1], linewidth=2)
    axes2[idx].set_title(f'{label} em cada Fold', fontsize=12, fontweight='bold')
    axes2[idx].set_xlabel('Fold', fontsize=10)
    axes2[idx].set_ylabel(label, fontsize=10)
    axes2[idx].set_xticks(df_compare['fold'])
    axes2[idx].legend()
    axes2[idx].grid(axis='both', linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

#%%
# Bland-Altman Plot
fig3, axes3 = plt.subplots(2, 2, figsize=(14, 10))
axes3 = axes3.flatten()

for idx, (metric, label) in enumerate(stats_metricas.items()):
    mean_of_both = (df_compare[f'{metric}_rf'] + df_compare[f'{metric}_xgb']) / 2
    diff = df_compare[f'{metric}_xgb'] - df_compare[f'{metric}_rf']
    mean_diff = diff.mean()
    std_diff = diff.std()
    
    axes3[idx].scatter(mean_of_both, diff, color='purple', alpha=0.7)
    for x, y, fold in zip(mean_of_both, diff, df_compare['fold']):
        axes3[idx].annotate(str(int(fold)), (x, y), textcoords='offset points',
                            xytext=(5, 4), fontsize=8, color='purple')
    axes3[idx].axhline(mean_diff, color='green', linestyle='--',
                       label=f'Viés: {mean_diff:.4f}')
    axes3[idx].axhline(mean_diff + 1.96*std_diff, color='red', linestyle=':', 
                       alpha=0.5, label='Limite de Conformidade (+)')
    axes3[idx].axhline(mean_diff - 1.96*std_diff, color='red', linestyle=':', 
                       alpha=0.5, label='Limite de Conformidade (-)')
    axes3[idx].set_title(f'{label}\n', 
                         fontsize=12, fontweight='bold')
    axes3[idx].set_xlabel(f'Média da Performance ({label})', fontsize=10)
    axes3[idx].set_ylabel(f'Diff (XGB - RF)', fontsize=10)
    axes3[idx].legend(loc='best', fontsize=8)
    axes3[idx].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()
# %%

# %% [markdown]
# Concordância de Importância de Features
BASE_DIR = Path(__file__).resolve().parent.parent
RF_RESULTS = os.path.join(BASE_DIR, "folds", "grid_search_rf")
XGB_RESULTS = os.path.join(BASE_DIR, "folds", "grid_search_xgb")

RF_FI_PATH = os.path.join(RF_RESULTS , "rf_feature_importance.csv")
XGB_FI_PATH = os.path.join(XGB_RESULTS, "xgb_feature_importance.csv") 

df_fi_rf = pd.read_csv(RF_FI_PATH)
df_fi_xgb = pd.read_csv(XGB_FI_PATH)

df_fi_rf.columns = [c.lower().strip() for c in df_fi_rf.columns]
df_fi_xgb.columns = [c.lower().strip() for c in df_fi_xgb.columns]

df_compare_fi = pd.merge(df_fi_rf, df_fi_xgb, on='feature', suffixes=('_rf', '_xgb'))

print(f"Features analisadas: {len(df_compare_fi)}")
display(df_compare_fi.sort_values(by='importance_mean_rf', ascending=False).head(50))

# T-TEST

# H0: A importância média dada pelo RF é igual à do XGB
t_stat_fi, p_val_fi = stats.ttest_rel(df_compare_fi['importance_mean_rf'], df_compare_fi['importance_mean_xgb'])

print(f"T-statistic: {t_stat_fi:.4f}")
print(f"P-value:     {p_val_fi:.4e}")

if p_val_fi < 0.05:
    print("Importâncias divergentes.")
else:
    print("Importâncias equivalentes.")

# SPEARMAN CORRELATION

# H0: Não há correlação entre os rankings
corr, p_spearman = stats.spearmanr(df_compare_fi['importance_mean_rf'], df_compare_fi['importance_mean_xgb'])

print(f"Coeficiente de Spearman: {corr:.4f} (-1 a +1)")
print(f"P-value:                {p_spearman:.4e}")

if p_spearman < 0.05:
    if corr > 0.7:
        print("Alta comcordancia.")
    elif 0.4 < corr <= 0.7:
        print("Moderada comcordancia.")
    else:
        print("Rankings divergentes.")
else:
    print("Correlação não significativa.")

#%%
df_compare_fi['rank_rf'] = df_compare_fi['importance_mean_rf'].rank(ascending=False)
df_compare_fi['rank_xgb'] = df_compare_fi['importance_mean_xgb'].rank(ascending=False)

plt.figure(figsize=(10, 6))
plt.scatter(df_compare_fi['rank_rf'], df_compare_fi['rank_xgb'], alpha=0.6, edgecolors='w', s=80)

lims = [1, len(df_compare_fi)]
plt.plot(lims, lims, 'r--', alpha=0.75, zorder=0, label='Concordância Perfeita')

top_features = df_compare_fi.nsmallest(48, 'rank_rf')['feature']
for feature in top_features:
    row = df_compare_fi[df_compare_fi['feature'] == feature].iloc[0]
    plt.annotate(row['feature'],
                 (row['rank_rf'], row['rank_xgb']),
                 xytext=(5, 5), textcoords='offset points', fontsize=9, color='black')

plt.xlabel('Rank RF (1 = mais importante)', fontsize=12)
plt.ylabel('Rank XGBoost (1 = mais importante)', fontsize=12)
plt.title('Concordância de Feature Importance por Ranking\n(Spearman r={:.2f})'.format(corr), fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.gca()
plt.gca()
plt.show()

# %%
