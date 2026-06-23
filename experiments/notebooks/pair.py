# %% [Markdown] COMPARAÇÃO ESTATÍSTICA: Grid Search vs Bayesian Optimization
# Random Forest, XGBoost, LightGBM
# Testes: Wilcoxon, t-test pareado, Friedman, Post-hoc Bonferroni, Bootstrap CI, Cohen's d

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# Grid Search 
RF_GRID_PATH = "../folds/grid_search_rf/Evaluation_rf_results.csv"
XGB_GRID_PATH = "../folds/grid_search_xgb/xgb_final_evaluation.csv"
LGB_GRID_PATH = "../folds/grid_search_lgb/lgb_final_evaluation.csv"

# Bayesian Optimization 
RF_BAY_PATH = "../folds/bayesian_rf/bayesian_rf_evaluation.csv"
XGB_BAY_PATH = "../folds/bayesian_xgb/bayesian_xgb_evaluation.csv"
LGB_BAY_PATH = "../folds/bayesian_lgb/bayesian_lgb_evaluation.csv"

METRICS = ['roc_auc', 'auprc', 'f1', 'precision', 'recall']
ALPHA = 0.05
N_BOOTSTRAP = 10000
SEED = 42

# ============================================================
# CARREGAMENTO
# ============================================================
rf_grid = pd.read_csv(RF_GRID_PATH)
xgb_grid = pd.read_csv(XGB_GRID_PATH)
lgb_grid = pd.read_csv(LGB_GRID_PATH)
rf_bay = pd.read_csv(RF_BAY_PATH)
xgb_bay = pd.read_csv(XGB_BAY_PATH)
lgb_bay = pd.read_csv(LGB_BAY_PATH)

all_configs = {
    'RF_Grid': rf_grid, 'RF_Bayesian': rf_bay,
    'XGB_Grid': xgb_grid, 'XGB_Bayesian': xgb_bay,
    'LGB_Grid': lgb_grid, 'LGB_Bayesian': lgb_bay
}

# Melhor configuração de cada modelo (baseado nos resultados)
best_configs = {
    'RF': rf_grid,      # Grid venceu
    'XGB': xgb_grid,    # Grid venceu
    'LGB': lgb_bay      # Bayesian venceu
}

# %% ============================================================
# 1. ESTATÍSTICAS DESCRITIVAS
# ============================================================
print("=" * 100)
print("1. ESTATÍSTICAS DESCRITIVAS")
print("=" * 100)

desc_rows = []
for name, df in all_configs.items():
    row = {'Config': name}
    for m in METRICS:
        row[f'{m}_mean'] = df[m].mean()
        row[f'{m}_std'] = df[m].std()
        row[f'{m}_median'] = df[m].median()
        row[f'{m}_min'] = df[m].min()
        row[f'{m}_max'] = df[m].max()
        row[f'{m}_cv'] = df[m].std() / df[m].mean() * 100
    desc_rows.append(row)

desc_df = pd.DataFrame(desc_rows)

for m in ['f1', 'roc_auc']:
    print(f"\n  {m.upper()}:")
    print(f"  {'Config':18s} {'Mean':>8s} {'Std':>8s} {'Median':>8s} {'Min':>8s} {'Max':>8s} {'CV%':>8s}")
    print(f"  {'-'*66}")
    for _, r in desc_df.iterrows():
        print(f"  {r['Config']:18s} {r[f'{m}_mean']:8.4f} {r[f'{m}_std']:8.4f} "
              f"{r[f'{m}_median']:8.4f} {r[f'{m}_min']:8.4f} {r[f'{m}_max']:8.4f} {r[f'{m}_cv']:8.2f}")

# %% ============================================================
# 2. TESTES PAREADOS: Grid vs Bayesian (por modelo)
# ============================================================
print(f"\n\n{'=' * 100}")
print("2. TESTES PAREADOS: Grid vs Bayesian (mesmo modelo, mesmos folds)")
print("   Wilcoxon signed-rank (não-paramétrico) + t-test pareado + Cohen's d")
print("=" * 100)

paired_results = []

for model, grid, bay in [("RF", rf_grid, rf_bay),
                          ("XGB", xgb_grid, xgb_bay),
                          ("LGB", lgb_grid, lgb_bay)]:
    print(f"\n  {model}:")
    for m in METRICS:
        g_vals = grid[m].values
        b_vals = bay[m].values
        diff = b_vals - g_vals

        # Wilcoxon signed-rank test
        try:
            w_stat, w_p = stats.wilcoxon(b_vals, g_vals)
        except Exception:
            w_stat, w_p = np.nan, np.nan

        # Paired t-test
        t_stat, t_p = stats.ttest_rel(b_vals, g_vals)

        # Cohen's d (pareado)
        d_mean = diff.mean()
        d_std = diff.std()
        cohens_d = d_mean / d_std if d_std > 0 else 0

        winner = "Bay" if d_mean > 0 else "Grid"
        sig_w = "*" if w_p < ALPHA else " "
        sig_t = "*" if t_p < ALPHA else " "

        paired_results.append({
            'model': model, 'metric': m, 'delta': d_mean,
            'wilcoxon_p': w_p, 'ttest_p': t_p, 'cohens_d': cohens_d
        })

        print(f"    {m:12s}: Δ={d_mean:+.4f} ({winner:4s}) | "
              f"Wilcoxon p={w_p:.4f}{sig_w} | t-test p={t_p:.4f}{sig_t} | "
              f"Cohen's d={cohens_d:+.3f}")

# %% ============================================================
# 3. TESTE DE FRIEDMAN + POST-HOC
# ============================================================
print(f"\n\n{'=' * 100}")
print("3. TESTE DE FRIEDMAN (k=3 melhores modelos, 10 folds pareados)")
print("   + Post-hoc Wilcoxon pairwise com correção de Bonferroni")
print("=" * 100)

print(f"\n  Friedman (H0: todas as distribuições são iguais):")
friedman_results = {}
for m in METRICS:
    vals = [best_configs[model][m].values for model in ['RF', 'XGB', 'LGB']]
    chi2, p_fried = stats.friedmanchisquare(*vals)
    sig = "***" if p_fried < 0.001 else "**" if p_fried < 0.01 else "*" if p_fried < 0.05 else "ns"
    friedman_results[m] = {'chi2': chi2, 'p': p_fried}
    print(f"    {m:12s}: χ²={chi2:.3f}, p={p_fried:.4f} {sig}")

# Post-hoc
pairs = [('RF', 'XGB'), ('RF', 'LGB'), ('XGB', 'LGB')]
n_comparisons = len(pairs)

print(f"\n  Post-hoc Wilcoxon (Bonferroni α_adj = {ALPHA/n_comparisons:.4f}):")
posthoc_results = []

for m in METRICS:
    print(f"\n    {m.upper()}:")
    for m1, m2 in pairs:
        v1 = best_configs[m1][m].values
        v2 = best_configs[m2][m].values
        diff_mean = v2.mean() - v1.mean()

        try:
            stat, p_raw = stats.wilcoxon(v1, v2)
        except Exception:
            stat, p_raw = np.nan, 1.0

        p_adj = min(p_raw * n_comparisons, 1.0)  # Bonferroni
        sig = "*" if p_adj < ALPHA else " "
        winner = m2 if diff_mean > 0 else m1

        posthoc_results.append({
            'metric': m, 'pair': f"{m1} vs {m2}",
            'delta': diff_mean, 'p_raw': p_raw, 'p_adj': p_adj
        })

        print(f"      {m1} vs {m2}: Δ={diff_mean:+.4f} ({winner:3s} ↑) | "
              f"p_raw={p_raw:.4f} | p_adj={p_adj:.4f}{sig}")

# %% ============================================================
# 4. RANKING MÉDIO POR FOLD
# ============================================================
print(f"\n\n{'=' * 100}")
print("4. RANKING MÉDIO POR FOLD (1=melhor em cada fold)")
print("=" * 100)

for m in METRICS:
    print(f"\n  {m.upper()}:")
    fold_data = pd.DataFrame({name: df[m].values for name, df in all_configs.items()})
    ranks = fold_data.rank(axis=1, ascending=False)
    mean_ranks = ranks.mean().sort_values()

    for config in mean_ranks.index:
        bar = "█" * int(mean_ranks[config] * 3)
        print(f"    {config:14s}: rank médio = {mean_ranks[config]:.2f} {bar}")

# %% ============================================================
# 5. BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================
print(f"\n\n{'=' * 100}")
print(f"5. BOOTSTRAP 95% CONFIDENCE INTERVALS (F1, n={N_BOOTSTRAP})")
print("=" * 100)

np.random.seed(SEED)

bootstrap_ci = {}
for name, df in all_configs.items():
    f1_vals = df['f1'].values
    boot_means = np.array([
        np.mean(np.random.choice(f1_vals, size=len(f1_vals), replace=True))
        for _ in range(N_BOOTSTRAP)
    ])
    ci_low = np.percentile(boot_means, 2.5)
    ci_high = np.percentile(boot_means, 97.5)
    bootstrap_ci[name] = (ci_low, f1_vals.mean(), ci_high)
    print(f"  {name:18s}: {f1_vals.mean():.4f} [{ci_low:.4f}, {ci_high:.4f}]")

# Bootstrap da diferença: melhor vs segundo melhor
print(f"\n  Diferença LGB_Bayesian - XGB_Grid (1o vs 2o):")
lgb_vals = lgb_bay['f1'].values
xgb_vals = xgb_grid['f1'].values
boot_diffs = np.array([
    np.mean(np.random.choice(lgb_vals, size=len(lgb_vals), replace=True)) -
    np.mean(np.random.choice(xgb_vals, size=len(xgb_vals), replace=True))
    for _ in range(N_BOOTSTRAP)
])
ci_low_d = np.percentile(boot_diffs, 2.5)
ci_high_d = np.percentile(boot_diffs, 97.5)
prob_superior = np.mean(boot_diffs > 0) * 100
print(f"  Δ médio = {boot_diffs.mean():+.4f} [{ci_low_d:+.4f}, {ci_high_d:+.4f}]")
print(f"  P(LGB_Bayesian > XGB_Grid) = {prob_superior:.1f}%")

# %% 


configs_list = list(all_configs.keys())
effect_matrix = np.zeros((len(configs_list), len(configs_list)))

print(f"\n  {'':18s}", end="")
for c in configs_list:
    print(f" {c:>12s}", end="")
print()

for i, c1 in enumerate(configs_list):
    print(f"  {c1:18s}", end="")
    for j, c2 in enumerate(configs_list):
        if i == j:
            print(f" {'---':>12s}", end="")
        else:
            v1 = all_configs[c1]['f1'].values
            v2 = all_configs[c2]['f1'].values
            pooled = np.sqrt((v1.std()**2 + v2.std()**2) / 2)
            d = (v2.mean() - v1.mean()) / pooled if pooled > 0 else 0
            effect_matrix[i, j] = d
            label = "~" if abs(d) < 0.2 else "S" if abs(d) < 0.5 else "M" if abs(d) < 0.8 else "L"
            print(f"  {d:+.2f}({label})", end="  ")
    print()

# %% 
fig = plt.figure(figsize=(20, 22))
gs = gridspec.GridSpec(3, 2, hspace=0.35, wspace=0.3)

colors_grid = {'RF_Grid': '#888780', 'XGB_Grid': '#888780', 'LGB_Grid': '#888780'}
colors_bay = {'RF_Bayesian': '#534AB7', 'XGB_Bayesian': '#534AB7', 'LGB_Bayesian': '#534AB7'}
colors_all = {**colors_grid, **colors_bay}
model_colors = {'RF_Grid': '#888780', 'RF_Bayesian': '#AFA9EC',
                'XGB_Grid': '#378ADD', 'XGB_Bayesian': '#85B7EB',
                'LGB_Grid': '#1D9E75', 'LGB_Bayesian': '#5DCAA5'}

# F1 por Fold (Grid vs Bayesian, 3 modelos) 
ax1 = fig.add_subplot(gs[0, 0])
folds = range(1, 11)
for name, df, color, ls in [
    ('RF Grid', rf_grid, '#888780', '-'), ('RF Bayesian', rf_bay, '#888780', '--'),
    ('XGB Grid', xgb_grid, '#378ADD', '-'), ('XGB Bayesian', xgb_bay, '#378ADD', '--'),
    ('LGB Grid', lgb_grid, '#1D9E75', '-'), ('LGB Bayesian', lgb_bay, '#1D9E75', '--')]:
    ax1.plot(folds, df['f1'].values, marker='o', markersize=4, label=name, color=color, linestyle=ls, linewidth=1.5)
ax1.set_xlabel('Fold')
ax1.set_ylabel('F1 Score')
ax1.set_title('F1 por Fold: Grid (sólido) vs Bayesian (tracejado)', fontweight='bold')
ax1.legend(fontsize=8, ncol=2, loc='lower left')
ax1.grid(alpha=0.3)
ax1.set_xticks(folds)

# Bootstrap CI 
ax2 = fig.add_subplot(gs[0, 1])
names_sorted = sorted(bootstrap_ci.keys(), key=lambda x: bootstrap_ci[x][1])
y_pos = range(len(names_sorted))
for i, name in enumerate(names_sorted):
    ci_low, mean, ci_high = bootstrap_ci[name]
    color = model_colors[name]
    ax2.barh(i, mean - ci_low, left=ci_low, height=0.6, color=color, alpha=0.3)
    ax2.barh(i, ci_high - mean, left=mean, height=0.6, color=color, alpha=0.3)
    ax2.plot(mean, i, 'o', color=color, markersize=8, zorder=5)
    ax2.plot([ci_low, ci_high], [i, i], '-', color=color, linewidth=2, zorder=4)
ax2.set_yticks(y_pos)
ax2.set_yticklabels(names_sorted)
ax2.set_xlabel('F1 Score')
ax2.set_title('Bootstrap 95% CI (F1)', fontweight='bold')
ax2.grid(alpha=0.3, axis='x')

# Paired Differences (Bayesian - Grid) 
ax3 = fig.add_subplot(gs[1, 0])
paired_df = pd.DataFrame(paired_results)
f1_paired = paired_df[paired_df['metric'] == 'f1']
x_pos = np.arange(len(f1_paired))
bar_colors = ['#E24B4A' if d < 0 else '#1D9E75' for d in f1_paired['delta']]
bars = ax3.bar(x_pos, f1_paired['delta'].values, color=bar_colors, alpha=0.8, edgecolor='white')
ax3.set_xticks(x_pos)
ax3.set_xticklabels(f1_paired['model'].values)
ax3.axhline(0, color='black', linewidth=0.8, linestyle='-')
ax3.set_ylabel('Δ F1 (Bayesian - Grid)')
ax3.set_title('Diferença F1: Bayesian vs Grid', fontweight='bold')
ax3.grid(alpha=0.3, axis='y')

# p-values
for i, (_, row) in enumerate(f1_paired.iterrows()):
    sig = "*" if row['wilcoxon_p'] < 0.05 else "ns"
    y_offset = 0.002 if row['delta'] > 0 else -0.004
    ax3.text(i, row['delta'] + y_offset, f"p={row['wilcoxon_p']:.3f}\n{sig}",
             ha='center', va='bottom' if row['delta'] > 0 else 'top', fontsize=9)

# Effect Size Heatmap 
ax4 = fig.add_subplot(gs[1, 1])
mask = np.eye(len(configs_list), dtype=bool)
effect_display = np.ma.masked_where(mask, effect_matrix)
im = ax4.imshow(effect_display, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
ax4.set_xticks(range(len(configs_list)))
ax4.set_xticklabels(configs_list, rotation=45, ha='right', fontsize=8)
ax4.set_yticks(range(len(configs_list)))
ax4.set_yticklabels(configs_list, fontsize=8)
ax4.set_title("Cohen's d Matrix (F1)", fontweight='bold')
plt.colorbar(im, ax=ax4, shrink=0.8, label="Cohen's d")

for i in range(len(configs_list)):
    for j in range(len(configs_list)):
        if i != j:
            d = effect_matrix[i, j]
            label = "~" if abs(d) < 0.2 else "S" if abs(d) < 0.5 else "M" if abs(d) < 0.8 else "L"
            color = 'white' if abs(d) > 0.8 else 'black'
            ax4.text(j, i, f"{d:+.1f}\n{label}", ha='center', va='center', fontsize=7, color=color)

# Friedman Rankings 
ax5 = fig.add_subplot(gs[2, 0])
rank_data = {}
for m in METRICS:
    fold_data = pd.DataFrame({name: df[m].values for name, df in all_configs.items()})
    ranks = fold_data.rank(axis=1, ascending=False)
    rank_data[m] = ranks.mean()

rank_df = pd.DataFrame(rank_data)
x = np.arange(len(METRICS))
width = 0.12
for i, config in enumerate(all_configs.keys()):
    vals = [rank_df.loc[config, m] for m in METRICS]
    ax5.bar(x + i * width, vals, width, label=config, color=model_colors[config], alpha=0.85)
ax5.set_xticks(x + width * 2.5)
ax5.set_xticklabels([m.upper() for m in METRICS])
ax5.set_ylabel('Rank Médio (menor = melhor)')
ax5.set_title('Ranking Médio por Métrica', fontweight='bold')
ax5.legend(fontsize=7, ncol=3)
ax5.grid(alpha=0.3, axis='y')
ax5.invert_yaxis()

# Box plot F1 
ax6 = fig.add_subplot(gs[2, 1])
f1_data = [df['f1'].values for df in all_configs.values()]
bp = ax6.boxplot(f1_data, labels=list(all_configs.keys()), patch_artist=True, widths=0.5)
for patch, color in zip(bp['boxes'], model_colors.values()):
    patch.set_facecolor(color)
    patch.set_alpha(0.6)
ax6.set_ylabel('F1 Score')
ax6.set_title('Distribuição F1 por Configuração', fontweight='bold')
ax6.tick_params(axis='x', rotation=30)
ax6.grid(alpha=0.3, axis='y')

plt.savefig('statistical_comparison.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()

# %% 
print(f"\n{'Config':18s} {'F1':>8s} {'95% CI':>20s} {'Rank':>6s} {'Melhor?':>10s}")

for name in ['LGB_Bayesian', 'XGB_Grid', 'RF_Grid', 'LGB_Grid', 'XGB_Bayesian', 'RF_Bayesian']:
    f1_mean = all_configs[name]['f1'].mean()
    ci_low, _, ci_high = bootstrap_ci[name]
    fold_data = pd.DataFrame({n: df['f1'].values for n, df in all_configs.items()})
    rank = fold_data.rank(axis=1, ascending=False).mean()[name]
    best = "★ BEST" if name == 'LGB_Bayesian' else ""
    print(f"  {name:16s} {f1_mean:8.4f} [{ci_low:.4f}, {ci_high:.4f}] {rank:6.2f} {best:>10s}")

'''print(f"""
CONCLUSÕES ESTATÍSTICAS:
  1. LGB_Bayesian é o melhor modelo (F1=0.850, rank=1.5)
  2. A melhoria do LGB_Bayesian sobre LGB_Grid é significativa
     (Wilcoxon p=0.002, Cohen's d=2.53, efeito GRANDE)
  3. RF e XGB não se beneficiaram da Bayesian Optimization
     (p>0.05 em F1, convergência tardia, sub-regularização)
  4. LGB_Bayesian vs XGB_Grid: P(LGB>XGB)={prob_superior:.1f}% via bootstrap,
     mas IC da diferença inclui zero — vantagem provável mas não conclusiva
  5. RF_Grid vs XGB_Grid: diferença negligível (Cohen's d=0.16)
""")'''