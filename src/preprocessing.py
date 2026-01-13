#%%
import numpy as np
import pandas as pd
import scipy.signal as signal
from scipy.ndimage import gaussian_filter1d
from src.config import (
    SAMPLING_RATES, 
    EDA_SMOOTH_SIGMA_MS, 
    BVP_FILTER_ORDER, 
    BVP_FILTER_RS, 
    BVP_PASSBAND
)

class SignalPreprocessor:
    def __init__(self):
        self.fs_bvp = SAMPLING_RATES.get('BVP', 64)
        self.fs_eda = SAMPLING_RATES.get('EDA', 4)

    def apply_eda_filter(self, eda_series):
        """
        Aplica filtro Gaussiano no sinal EDA para reduzir ruído de alta frequência.
        """
        sigma_samples = (EDA_SMOOTH_SIGMA_MS / 1000) * self.fs_eda
        # Ensure we have a 1D array - flatten if needed
        eda_values = eda_series.values.ravel()
        eda_clean_values = gaussian_filter1d(eda_values, sigma=sigma_samples)
        return pd.Series(eda_clean_values, index=eda_series.index, name='EDA_Clean')

    def compute_acc_magnitude(self, acc_df):
        """
        Calcula a magnitude vetorial do acelerômetro: sqrt(x^2 + y^2 + z^2).
        """
        # Garante numérico
        for col in ['x', 'y', 'z']:
            if col in acc_df.columns:
                 acc_df[col] = pd.to_numeric(acc_df[col], errors='coerce').fillna(0)
        
        magnitude = np.sqrt(acc_df['x']**2 + acc_df['y']**2 + acc_df['z']**2)
        
        # Suavização leve para remover jitter do sensor
        mag_smooth = gaussian_filter1d(magnitude, sigma=5)
        return pd.Series(mag_smooth, index=acc_df.index, name='ACC_Mag')

    def apply_bvp_filter(self, bvp_series):
        """
        Filtro passa-banda (Chebyshev II) para limpar o sinal de BVP.
        Útil se formos extrair HRV depois, mas opcional para o dataset mestre de 4Hz.
        """
        if bvp_series.empty: return bvp_series
        
        # Preenche buracos
        bvp_series = bvp_series.ffill().bfill()
        
        nyquist = 0.5 * self.fs_bvp
        low = BVP_PASSBAND[0] / nyquist
        high = BVP_PASSBAND[1] / nyquist
        
        # Evita erro se frequencias forem invalidas
        if low <= 0 or high >= 1:
             return bvp_series

        sos = signal.cheby2(
            N=BVP_FILTER_ORDER, 
            rs=BVP_FILTER_RS, 
            Wn=[low, high], 
            btype='bandpass', 
            output='sos'
        )
        bvp_clean_values = signal.sosfiltfilt(sos, bvp_series.values)
        return pd.Series(bvp_clean_values, index=bvp_series.index, name='BVP_Clean')

    def synchronize_data(self, data_bundle, target_fs=4):
        """
        Sincroniza todos os sinais para uma frequência comum (padrão 4Hz).
        Gera um DataFrame contínuo para análise exploratória.
        """
        # Sensores principais para EDA
        # Nota: Usamos HR (já processado) em vez de BVP bruto para correlação visual simples
        desired_sensors = ['EDA', 'TEMP', 'HR'] 
        
        # Verifica quais existem
        valid_sensors = [s for s in desired_sensors if s in data_bundle and not data_bundle[s].empty]
        
        if not valid_sensors:
            return pd.DataFrame()

        # 1. Definir intervalo de tempo comum (Interseção)
        start_time = max(data_bundle[s].index[0] for s in valid_sensors)
        end_time = min(data_bundle[s].index[-1] for s in valid_sensors)
        
        # Se ACC existe, usa ele também para limitar o tempo
        if 'ACC' in data_bundle:
            start_time = max(start_time, data_bundle['ACC'].index[0])
            end_time = min(end_time, data_bundle['ACC'].index[-1])

        # Cria index mestre (Ex: 4Hz = 250ms)
        freq_str = f"{int(1000/target_fs)}ms"
        master_index = pd.timedelta_range(start=start_time, end=end_time, freq=freq_str)
        
        aligned_data = []

        # --- Processa EDA ---
        if 'EDA' in data_bundle:
            eda_raw = data_bundle['EDA']
            # Aplica filtro antes de resample (melhor qualidade)
            eda_clean = self.apply_eda_filter(eda_raw)
            # Reindexa e interpola
            eda_aligned = eda_clean.reindex(master_index, method='nearest').interpolate()
            aligned_data.append(eda_aligned)

        # --- Processa TEMP ---
        if 'TEMP' in data_bundle:
            temp = data_bundle['TEMP']
            temp_aligned = temp.reindex(master_index, method='nearest').interpolate()
            temp_aligned.name = 'TEMP'
            aligned_data.append(temp_aligned)

        # --- Processa HR (Heart Rate) ---
        if 'HR' in data_bundle:
            hr = data_bundle['HR']
            # HR é amostrado a 1Hz originalmente, fazemos upsample suave para 4Hz
            hr_aligned = hr.reindex(master_index).interpolate(method='time')
            hr_aligned.name = 'HR'
            aligned_data.append(hr_aligned)

        # --- Processa ACC (Calcula Magnitude -> Downsample) ---
        if 'ACC' in data_bundle:
            acc_raw = data_bundle['ACC']
            # Calcula magnitude na frequencia original (32Hz) para não perder picos de movimento
            acc_mag = self.compute_acc_magnitude(acc_raw)
            # Resample usando média para baixar para 4Hz
            acc_aligned = acc_mag.resample(freq_str).mean()
            # Alinha com master index
            acc_aligned = acc_aligned.reindex(master_index).interpolate()
            aligned_data.append(acc_aligned)

        if not aligned_data:
            return pd.DataFrame()

        # Junta tudo
        master_df = pd.concat(aligned_data, axis=1)
        
        # Remove linhas que sobraram com NaN nas bordas
        master_df = master_df.dropna()
        
        return master_df
    
#%%
