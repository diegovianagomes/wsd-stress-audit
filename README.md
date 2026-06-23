# WSD — Detecção de Estresse com Wearables

Projeto da disciplina **Tópicos em Inteligência Artificial e Otimização I (PCC557)** : Prof. Gustavo Andrade do Vale.

## O que é

Estudo de **detecção de estresse a partir de sinais fisiológicos** captados por um wearable de pulso (Empatica E4), usando o dataset Wearable Device Dataset from Induced Stress and Structured Exercise Sessions (Hongn et al., 2025 — PhysioNet / *Scientific Data* 12:520), com 36 sujeitos adultos saudáveis submetidos a protocolos de estresse cognitivo, exercício aeróbico e anaeróbico.

A partir dos sinais brutos (EDA, BVP/HR, HRV, temperatura e acelerometria), o projeto extrai 49 features em janelas de 60s e treina modelos de classificação (Random Forest, XGBoost, LightGBM) sob **validação honesta por sujeito** (GroupKFold), evitando vazamento entre treino e teste.

## Achado principal

Sob validação independente por sujeito, **nenhuma técnica de balanceamento, normalização ou redução de features supera de forma relevante o baseline**, e o desempenho agregado modesto (AUC ≈ 0,64) esconde uma distribuição **bimodal entre sujeitos**. A análise por sujeito, cruzada com a autoavaliação de estresse do protocolo, revela uma **dissociação entre o estresse subjetivo relatado e o sinal autonômico periférico** captado pelo wearable: para parte dos indivíduos a fisiologia periférica se move na direção contrária ao padrão canônico de estresse. Isso torna o rótulo binário inadequado por construção, e não por ruído.




Os scripts em `scripts/` são numerados na ordem do pipeline: comece pela extração de features (`01_stress.py` / `01_exercise.py ...`) e depois o treino (`03_train_nested_cv.py`). 


## Dataset
Hongn, A., Bosch, F., Prado, L.E. et al. Wearable Physiological Signals under Acute Stress and Exercise Conditions. Sci Data 12, 520 (2025). https://doi.org/10.1038/s41597-025-04845-9
