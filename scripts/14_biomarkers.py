#%% 
# Recomputa, para o conjunto corrigido de dissociados identificado pelo pipeline
# subject-independent (GroupKFold), os marcadores autonomicos em repouso e a
# reatividade, expressos como z-score relativo a coorte. O objetivo e descrever
# a assinatura fisiologica de cada sujeito de forma quantitativa e neutra, sem
# rotulo diagnostico.
#
# Marcadores em repouso (REST):
#   mean_tonic_eda : tonus eletrodermico basal (simpatico)
#   rmssd, pnn50   : variabilidade cardiaca basal (parassimpatico)
#   ratio          : LF/HF basal (balanco simpatovagal)
#   hr_mean        : frequencia cardiaca basal
#   acc_std        : movimento basal
# Reatividade (STRESS - REST por sujeito):
#   scr_mean_amp, mean_raw_eda : resposta eletrodermica
#   rmssd, pnn50, hr_mean      : resposta cardiaca
#
# z = (valor do sujeito - media dos demais) / desvio dos demais, calculado sobre
# as MEDIAS POR SUJEITO e excluindo os alvos da referencia, de modo que o alvo
# nao infle a propria base de comparacao. Sem corte fixo: o z mostra a magnitude.


import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# CONFIGURACAO


def _find_project_root(marker="src"):
    try:
        start = Path(__file__).resolve()
    except NameError:
        start = Path.cwd().resolve()
    for parent in [start, *start.parents]:
        if (parent / marker).is_dir():
            return parent
    raise FileNotFoundError(f"Nope'{marker}').")

PROJECT_ROOT = _find_project_root("src")
sys.path.insert(0, str(PROJECT_ROOT))
from src.config import PROCESSED_DIR

FIG_DIR = PROJECT_ROOT / "experiments" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "label"
GROUP = "subject_id"
TARGETS = ['f02', 'f03', 'f04', 'f18']
REST_MARKERS = ['mean_tonic_eda', 'rmssd', 'pnn50', 'ratio', 'hr_mean', 'acc_std']
REACT_FEATS = ['scr_mean_amp', 'mean_raw_eda', 'rmssd', 'pnn50', 'hr_mean']
REPORT_REST_COLS = ['Baseline', 'First Rest', 'Second Rest']
HIGHLIGHT = 1.5


# %% MARCADORES AUTONOMICOS DOS DISSOCIADOS (f02, f03, f04, f18)

data = pd.read_csv(Path(PROCESSED_DIR) / "dataset_stress.csv")
feat_cols = [c for c in data.columns if c not in
             ['subject_id', 'window_id', 'label', 'scenario', 'protocol']]
REST_MARKERS = [f for f in REST_MARKERS if f in feat_cols]
REACT_FEATS = [f for f in REACT_FEATS if f in feat_cols]
present = [s for s in TARGETS if s in data[GROUP].unique()]
n_others = data.loc[~data[GROUP].isin(TARGETS), GROUP].nunique()

print(f"\nJanelas {len(data)} | sujeitos {data[GROUP].nunique()} | "
      f"alvos presentes {present} | amostra de referencia {n_others} sujeitos")

rest = data[data[TARGET] == 0]
stress = data[data[TARGET] == 1]


def per_subj_mean(df, feat):
    return df.groupby(GROUP)[feat].mean()


def zframe(frame):
    """z de cada coluna relativo aos demais sujeitos (alvos fora da referencia)."""
    others = frame[~frame.index.isin(TARGETS)]
    mu, sd = others.mean(), others.std()
    z = (frame - mu) / sd
    return z, mu, sd


# tamanhos por alvo
sizes = pd.DataFrame({
    'n_rest': rest.groupby(GROUP).size(),
    'n_stress': stress.groupby(GROUP).size(),
}).loc[present]


# %% 1. MARCADORES EM REPOUSO (valor e z relativo a coorte)


rest_raw = pd.DataFrame({m: per_subj_mean(rest, m) for m in REST_MARKERS})
rest_z, rest_mu, rest_sd = zframe(rest_raw)

ref = pd.DataFrame({'amostra_media': rest_mu, 'amostra_dp': rest_sd}).T
print(pd.concat([rest_raw.loc[present], ref]).round(3).to_string())
print(rest_z.loc[present].round(2).to_string())


# %% 2. REATIVIDADE (STRESS - REST), valor e z

delta = pd.DataFrame({m: (per_subj_mean(stress, m) - per_subj_mean(rest, m))
                      for m in REACT_FEATS})
react_z, react_mu, react_sd = zframe(delta)


refd = pd.DataFrame({'amostra_media': react_mu, 'amostra_dp': react_sd}).T
print(pd.concat([delta.loc[present], refd]).round(3).to_string())
print(react_z.loc[present].round(2).to_string())


# %% 3. AUTORRELATO EM REPOUSO

sr_files = list(PROJECT_ROOT.rglob("Stress_Level_v*.csv"))
sr = {}
for f in sr_files:
    try:
        df = pd.read_csv(f, index_col=0)
    except Exception:
        continue
    cols = [c for c in REPORT_REST_COLS if c in df.columns]
    if not cols:
        continue
    for s in df.index:
        sr[s] = float(df.loc[s, cols].mean())

for s in present:
    val = sr.get(s, np.nan)
    print(f"  {s}: autorrelato em repouso = "
          f"{val:.2f}/10" if not np.isnan(val) else f"  {s}: nao disponivel")

# %% ASSINATURA AUTONOMICA (heatmap de z)

nice_rest = {'mean_tonic_eda': 'EDAton', 'rmssd': 'RMSSD', 'pnn50': 'pNN50',
             'ratio': 'LF/HF', 'hr_mean': 'HR', 'acc_std': 'ACC'}
nice_react = {'scr_mean_amp': 'dSCR', 'mean_raw_eda': 'dEDA', 'rmssd': 'dRMSSD',
              'pnn50': 'dpNN50', 'hr_mean': 'dHR'}

rz = rest_z.loc[present, REST_MARKERS].rename(columns=nice_rest)
kz = react_z.loc[present, REACT_FEATS].rename(columns=nice_react)
combo = pd.concat([rz, kz], axis=1)

fig, ax = plt.subplots(figsize=(0.85 * combo.shape[1] + 2, 0.7 * len(present) + 2))
im = ax.imshow(combo.to_numpy(), cmap='RdBu_r', vmin=-3, vmax=3, aspect='auto')
ax.set_xticks(range(combo.shape[1]))
ax.set_xticklabels(combo.columns, rotation=40, ha='right', fontsize=9)
ax.set_yticks(range(len(present)))
ax.set_yticklabels(present)
ax.axvline(len(REST_MARKERS) - 0.5, color='k', lw=1.2)
ax.text(len(REST_MARKERS) / 2 - 0.5, -0.8, "repouso", ha='center', fontsize=9)
ax.text(len(REST_MARKERS) + len(REACT_FEATS) / 2 - 0.5, -0.8, "reatividade",
        ha='center', fontsize=9)
for i in range(len(present)):
    for j in range(combo.shape[1]):
        v = combo.to_numpy()[i, j]
        ax.text(j, i, f"{v:+.1f}", ha='center', va='center', fontsize=8,
                color='white' if abs(v) > 1.8 else '#222')
ax.set_title("Assinatura autonomica por sujeito (z relativo a coorte)")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label='z')
plt.tight_layout()
fig_path = FIG_DIR / "assinatura_autonomica_dissociados.png"
plt.savefig(fig_path, dpi=150)
print(f"\nFigura salva em: {fig_path}")

print("\nScript 13 concluido.")