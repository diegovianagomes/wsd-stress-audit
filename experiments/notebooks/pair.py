#%%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

BASE_DIR = r"C:\DEV\mestrado\WSD\experiments"
RF_RESULTS = os.path.join(BASE_DIR, "folds", "grid_search", "Evaluation_results.csv")
XGB_RESULTS = os.path.join(BASE_DIR, "folds", "grid_search_xgb", "xgb_final_evaluation.csv")

def load_clean(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    return df

df_rf = load_clean(RF_RESULTS)
df_xgb = load_clean(XGB_RESULTS)

if df_rf.empty or df_xgb.empty:
    print("Pare a execução e verifique os caminhos.")
else:
    col_metric = 'f1' if 'f1' in df_rf.columns else ('f1-score' if 'f1-score' in df_rf.columns else 'roc_auc')
    
    df_comp = pd.merge(
        df_rf[['fold', col_metric]], 
        df_xgb[['fold', col_metric]], 
        on='fold', 
        suffixes=('_rf', '_xgb')
    )

    metric = col_metric
    df_comp['diff'] = df_comp[f'{metric}_xgb'] - df_comp[f'{metric}_rf']

    print(f"Estatística Descritiva ({metric.upper()})")
    summary = df_comp[[f'{metric}_rf', f'{metric}_xgb']].agg(['mean', 'std', 'median'])
    print(summary)

    mean_rf = summary.loc['mean', f'{metric}_rf']
    mean_xgb = summary.loc['mean', f'{metric}_xgb']
    ganho = ((mean_xgb - mean_rf) / mean_rf) * 100
    print(f"\nGanho Médio do XGBoost sobre RF: {ganho:.2f}%")

    _, p_norm = stats.shapiro(df_comp['diff'])

    if p_norm > 0.05:
        stat, p_val = stats.ttest_rel(df_comp[f'{metric}_xgb'], df_comp[f'{metric}_rf'])
        test_name = "Paired T-Test "
    else:
        stat, p_val = stats.wilcoxon(df_comp[f'{metric}_xgb'], df_comp[f'{metric}_rf'])
        test_name = "Wilcoxon Signed-Rank Test (Não-Paramétrico)"

    print(f"\nTeste Aplicado: {test_name}")
    print(f"P-Value: {p_val:.4f}")

    # BOXPLOT
    plt.figure(figsize=(10, 6))

    df_melt = df_comp.melt(id_vars='fold', 
                           value_vars=[f'{metric}_rf', f'{metric}_xgb'], 
                           var_name='Modelo', 
                           value_name=metric.upper()
    )
    df_melt['Modelo'] = df_melt['Modelo'].replace({f'{metric}_rf': 'Random Forest', f'{metric}_xgb': 'XGBoost'})
    
    order = ['Random Forest', 'XGBoost']
    
    sns.boxplot(x='Modelo', 
                y=metric.upper(), 
                data=df_melt, 
                palette=['#9b59b6', '#2ecc71'], 
                width=0.4, 
                order=order
    )
    
    # Pontos individuais
    sns.stripplot(
        x='Modelo', 
        y=metric.upper(), 
        data=df_melt, 
        color='black', 
        alpha=0.5, 
        size=8, 
        order=order
        )
    # Linhas
    for i in df_comp['fold']:
        y_rf = df_comp.loc[df_comp['fold'] == i, f'{metric}_rf'].values[0]
        y_xgb = df_comp.loc[df_comp['fold'] == i, f'{metric}_xgb'].values[0]
        
        plt.plot(
            [0, 1], 
            [y_rf, y_xgb], 
            color='gray', 
            linestyle='--', 
            alpha=0.4, 
            linewidth=1
        )

    plt.title(f'Comparação RF vs XGBoost ({metric.upper()})\n{test_name} (p={p_val:.4f})', 
              fontsize=14, fontweight='bold')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

    # DIFERENÇAS POOR FOLD
    plt.figure(figsize=(10, 5))
    cores = ['#2ecc71' if x >= 0 else '#e74c3c' for x in df_comp['diff']]

    ax = sns.barplot(x='fold', y='diff', data=df_comp, palette=cores)
    plt.axhline(0, color='black', linewidth=1)

    plt.title(f'Diferença de {metric.upper()} por Fold (XGBoost - Random Forest)', fontsize=14, fontweight='bold')
    plt.ylabel('Diferença (Delta)')
    plt.xlabel('Fold Externo')
    plt.grid(axis='y', alpha=0.3)

    for container in ax.containers:
        ax.bar_label(container, fmt='%.3f', padding=3)

    plt.tight_layout()
    plt.show()
# %%
