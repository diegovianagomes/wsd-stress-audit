# %% [markdown]
# # Processamento de Sinais de apenas um Sujeito


# %% 
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

current_dir = Path(os.getcwd())
project_root = current_dir.parent.parent

if str(project_root) not in sys.path:
    sys.path.append(str(project_root))

from src.config import RAW_DATA_DIR, PROCESSED_DIR
from src.data_loader import load_subject_raw_data
from src.preprocessing import SignalPreprocessor
from src.labeling import get_session_start_time, load_tags, apply_labels
from src.feature_extraction import FeatureExtractor

#%% [markdown] 
# Configurações do Experimento
SUBJECT_ID = "f01"
EXPERIMENT = "STRESS"

# Caminho dos dados
subject_path = RAW_DATA_DIR / EXPERIMENT / SUBJECT_ID

print(f"--> Processando Sujeito: {SUBJECT_ID}")
print(f"--> Caminho: {subject_path}")

# %% 2. Carregamento dos Dados Brutos
if not subject_path.exists():
    raise FileNotFoundError(f"Pasta não encontrada: {subject_path}")

# Carrega dicionário {'ACC': df, 'EDA': df...}
raw_data = load_subject_raw_data(subject_path)
print("Dados carregados:", raw_data.keys())

# %% 3. Sincronização e Rotulagem (Ground Truth)
# Precisamos alinhar os dados apenas para saber ONDE pintar o label 0 ou 1

# 3.1 Pega o início real da sessão (Data absoluta)
session_start = get_session_start_time(subject_path)
print(f"Início da Sessão (UTC): {session_start}")

# 3.2 Carrega as Tags e converte para segundos
tags_df = load_tags(subject_path, session_start)
print(f"Tags encontradas: {len(tags_df)}")

# 3.3 Gera um DataFrame contínuo (Timeline) para aplicar os labels
preprocessor = SignalPreprocessor()
df_timeline = preprocessor.synchronize_data(raw_data)

# 3.4 Aplica os Labels (Fases 1, 3, 4, 5 = Stress)
# df_labeled agora tem uma coluna 'label' contínua (0 ou 1)
df_labeled = apply_labels(df_timeline, tags_df, stress_phases=[1, 3, 4, 5])

# Validação Rápida
counts = df_labeled['label'].value_counts()
print("\nDistribuição de Tempo (4Hz):")
print(f"Repouso: {counts.get(0, 0)} amostras")
print(f"Stress:  {counts.get(1, 0)} amostras")

# %% 4. Extração de Features (Janelamento)
# Aqui transformamos o sinal contínuo em Tabela de Treino (X, y)

# Janela de 60s com passo de 30s (Sobreposição 50%)
extractor = FeatureExtractor(window_size=60, step_size=30)

print("\nIniciando extração de features...")
# Passamos raw_data (para precisão de cálculo) e df_labeled (para saber o target)
df_features = extractor.extract_features(raw_data, df_labeled)

# Remove linhas com NaNs (ex: janelas no início/fim sem dados suficientes)
df_features = df_features.dropna()

print(f"Dataset Final Gerado do Sujeito: {df_features.shape}")
df_features.head(100)

# %% 5. Visualização (Check de Sanidade)
# Plota a média do EDA de cada janela colorida pelo Label
plt.figure(figsize=(15, 5))
plt.scatter(df_features['window_id'], df_features['mean_raw_eda'], 
            c=df_features['label'], cmap='coolwarm', alpha=0.6)
plt.colorbar(label='Label (0=Blue, 1=Red)')
plt.title(f"Distribuição de Features (Janelas) - {SUBJECT_ID}")
plt.xlabel("Tempo da Janela (s)")
plt.ylabel("Média EDA (µS)")
plt.show()

# %% 6. Salvar
filename = f"{EXPERIMENT}_{SUBJECT_ID}_features.csv"
output_path = PROCESSED_DIR / filename
df_features.to_csv(output_path, index=False)

print(f"\nSalvo com sucesso em: {output_path}")