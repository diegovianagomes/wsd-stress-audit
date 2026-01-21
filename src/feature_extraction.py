import numpy as np
import pandas as pd
from scipy import signal
import neurokit2 as nk
from src.config import WINDOW_SIZE, STEP_SIZE


class FeatureExtractor:
    def __init__(self, window_size=WINDOW_SIZE, step_size=STEP_SIZE):
        self.window_size = window_size
        self.step_size = step_size

    def calculate_slope_ratios(self, series):
        if len(series) < 2: return 0.0, 0.0
        diff = np.diff(series)
        n = len(diff) if len(diff) > 0 else 1
        ratio_up = np.sum(diff > 0) / n
        ratio_down = np.sum(diff < 0) / n
        return ratio_up, ratio_down
    
    # BVP
    def extract_bvp_features(self, bvp_window):
        feats = {}
        if bvp_window.empty: return {'bvp_mean': 0, 'bvp_std': 0}
        vals = bvp_window.values.flatten()
        feats['bvp_mean'] = np.mean(vals)
        feats['bvp_std'] = np.std(vals)
        return feats

    # HR
    def extract_hr_features(self, hr_window):
        feats = {}
        if hr_window.empty: return {k: 0 for k in ['hr_mean', 'hr_std', 'hr_ratio_down', 'hr_ratio_up']}
        vals = hr_window.values.flatten()
        feats['hr_mean'] = np.mean(vals)
        feats['hr_std'] = np.std(vals)
        r_up, r_down = self.calculate_slope_ratios(vals)
        feats['hr_ratio_up'] = r_up
        feats['hr_ratio_down'] = r_down
        return feats

    # ACC
    def extract_acc_features(self, acc_window):
        
        keys = ['acc_x_mean', 
                'acc_x_std', 
                'acc_y_mean', 
                'acc_y_std', 
                'acc_z_mean', 
                'acc_z_std', 
                'acc_mean', 
                'acc_std', 
                'acc_ratio_up', 
                'acc_ratio_down'
        ]
        feats = {k: 0 for k in keys}
        
        if acc_window.empty: return feats
        
        for ax in ['x', 'y', 'z']:
            vals = acc_window[ax].values
            feats[f'acc_{ax}_mean'] = np.mean(vals)
            feats[f'acc_{ax}_std'] = np.std(vals)
        
        mag = np.sqrt(acc_window['x']**2 + acc_window['y']**2 + acc_window['z']**2).values
        feats['acc_mean'] = np.mean(mag)
        feats['acc_std'] = np.std(mag)
        r_up, r_down = self.calculate_slope_ratios(mag)
        feats['acc_ratio_up'] = r_up
        feats['acc_ratio_down'] = r_down
        return feats

    # EDA
    def extract_eda_features(self, eda_window, fs=4):
        
        defaults = {
            'mean_raw_eda': 0, 
            'std_raw_eda': 0,
            'mean_tonic_eda': 0, 
            'std_tonic_eda': 0,
            'mean_phasic_eda': 0, 
            'std_phasic_eda': 0,
            'tonic_ratio_up': 0, 
            'tonic_ratio_down': 0,
            'peaks_density': 0, 
            'scr_mean_amp': 0, 
            'scr_mean_height': 0,
            'scr_mean_risetime': 0, 
            'scr_mean_recoverytime': 0
        }
        if len(eda_window) < fs * 10: return defaults

        feats = {}
        vals = eda_window.values.flatten()
        feats['mean_raw_eda'] = np.mean(vals)
        feats['std_raw_eda'] = np.std(vals)
        
        try:
            eda_decomposed = nk.eda_phasic(vals, sampling_rate=fs, method='highpass')
            tonic = eda_decomposed['EDA_Tonic'].values
            phasic = eda_decomposed['EDA_Phasic'].values
            
            feats['mean_tonic_eda'] = np.mean(tonic)
            feats['std_tonic_eda'] = np.std(tonic)
            t_up, t_down = self.calculate_slope_ratios(tonic)
            feats['tonic_ratio_up'] = t_up
            feats['tonic_ratio_down'] = t_down
            
            feats['mean_phasic_eda'] = np.mean(phasic)
            feats['std_phasic_eda'] = np.std(phasic)
            
            peaks, info = nk.eda_peaks(phasic, sampling_rate=fs, amplitude_min=0.05)
            n_peaks = np.sum(peaks['SCR_Peaks'])
            feats['peaks_density'] = n_peaks / (len(vals)/fs) 
            
            if n_peaks > 0:
                feats['scr_mean_amp'] = np.nanmean(info.get('SCR_Amplitude', 0))
                feats['scr_mean_height'] = np.nanmean(info.get('SCR_Height', 0))
                feats['scr_mean_risetime'] = np.nanmean(info.get('SCR_RiseTime', 0))
                feats['scr_mean_recoverytime'] = np.nanmean(info.get('SCR_RecoveryTime', 0))
            else:
                feats['scr_mean_amp'] = 0; feats['scr_mean_height'] = 0
                feats['scr_mean_risetime'] = 0; feats['scr_mean_recoverytime'] = 0
                
        except: return defaults
        return feats

    #  HRV -> Usa LOMB-SCARGLE para calcular Frequência, pois não requer interpolação e é o método padrão para séries temporais irregulares como IBI.
    def extract_hrv_features(self, ibi_window):

        keys = ['max_ibi', 
                'min_ibi', 
                'mean_ibi', 
                'hr_mean_ibi', 
                'pnn20', 
                'pnn50', 
                'rmssd', 
                'sdnn',
                'total_power', 
                'ratio', 
                'VLF_power', 
                'VLF_peak', 
                'LF_power', 
                'LF_peak', 
                'LF_n',
                'HF_power', 
                'HF_peak', 
                'HF_n', 
                'VHF_power', 
                'VHF_peak']
        feats = {k: 0.0 for k in keys}
        
        if ibi_window.empty: return feats
        ibi_clean = ibi_window[(ibi_window > 0.3) & (ibi_window < 1.3)].dropna()
        if len(ibi_clean) < 5: return feats

        # Time Domain
        try:
            rri = ibi_clean.values * 1000.0
            diff_rri = np.diff(rri)
            
            feats['mean_ibi'] = np.mean(ibi_clean)
            feats['max_ibi'] = np.max(ibi_clean)
            feats['min_ibi'] = np.min(ibi_clean)
            if feats['mean_ibi'] > 0:
                feats['hr_mean_ibi'] = 60 / feats['mean_ibi']
            
            feats['sdnn'] = np.std(rri, ddof=1)
            feats['rmssd'] = np.sqrt(np.mean(diff_rri**2))
            
            if len(diff_rri) > 0:
                feats['pnn50'] = 100 * np.sum(np.abs(diff_rri) > 50) / len(diff_rri)
                feats['pnn20'] = 100 * np.sum(np.abs(diff_rri) > 20) / len(diff_rri)
        except:
            return feats

        # Frequency Domain
        try:
            # Tempo acumulado dos batimentos (Eixo X)
            t_rri = np.cumsum(rri) / 1000.0 
            t_rri = t_rri - t_rri[0]
            
            # Valores dos batimentos (Eixo Y)
            y_rri = rri - np.mean(rri)

            # Define as frequências que queremos analisar (0.01Hz até 0.5Hz)
            freqs = np.linspace(0.01, 1.0, 500) 
            
            # Calcula Periodograma Lomb-Scargle
            # angular frequencies = 2 * pi * freqs
            pgram = signal.lombscargle(t_rri, y_rri, freqs * 2 * np.pi, normalize=False)
            
            # O retorna potência não normalizada pela frequência
            psd = pgram * (1.0 / len(t_rri))

            # Função para extrair métricas da banda
            def get_band_metrics(f_arr, p_arr, low, high):
                mask = (f_arr >= low) & (f_arr < high)
                if not np.any(mask): return 0.0, 0.0
                
                # Potência = Área sob a curva 
                power = np.trapz(p_arr[mask], f_arr[mask])
                
                # Pico = Frequência onde a potência é máxima
                peak_idx = np.argmax(p_arr[mask])
                peak_freq = f_arr[mask][peak_idx]
                
                return power, peak_freq

            # Extração
            vlf_p, vlf_peak = get_band_metrics(freqs, psd, 0.003, 0.04)
            lf_p, lf_peak   = get_band_metrics(freqs, psd, 0.04, 0.15)
            hf_p, hf_peak   = get_band_metrics(freqs, psd, 0.15, 0.40)
            vhf_p, vhf_peak = get_band_metrics(freqs, psd, 0.40, 1.0)

            total_p = vlf_p + lf_p + hf_p + vhf_p
            
            feats['total_power'] = total_p
            feats['VLF_power'] = vlf_p; 
            feats['VLF_peak'] = vlf_peak
            feats['LF_power'] = lf_p;   
            feats['LF_peak'] = lf_peak
            feats['HF_power'] = hf_p;   
            feats['HF_peak'] = hf_peak
            feats['VHF_power'] = vhf_p; 
            feats['VHF_peak'] = vhf_peak
            
            # Ratio e Normalizados
            if hf_p > 0: 
                feats['ratio'] = lf_p / hf_p
            
            if total_p > 0:
                feats['LF_n'] = lf_p / total_p
                feats['HF_n'] = hf_p / total_p

        except Exception as e:
            pass

        for k in feats:
            if pd.isna(feats[k]) or np.isinf(feats[k]): feats[k] = 0.0
            
        return feats

    def extract_features(self, raw_data_dict, labels_df):
        acc_idx = raw_data_dict['ACC'].index
        start_time = acc_idx[0]
        end_time = acc_idx[-1]
        total_duration = (end_time - start_time).total_seconds()
        
        features_list = []
        
        for current_sec in np.arange(0, total_duration - self.window_size, self.step_size):
            t_start = start_time + pd.Timedelta(seconds=current_sec)
            t_end = start_time + pd.Timedelta(seconds=current_sec + self.window_size)
            
            acc_win = raw_data_dict['ACC'][t_start:t_end]
            
            bvp_win = raw_data_dict['BVP'][t_start:t_end] if 'BVP' in raw_data_dict else pd.DataFrame()
            eda_win = raw_data_dict['EDA'][t_start:t_end] if 'EDA' in raw_data_dict else pd.DataFrame()
            hr_win  = raw_data_dict['HR'][t_start:t_end]  if 'HR' in raw_data_dict else pd.DataFrame()

            ibi_win = pd.Series(dtype=float)
            if 'IBI' in raw_data_dict:
                ibi_slice = raw_data_dict['IBI'][t_start:t_end]
                if not ibi_slice.empty: ibi_win = ibi_slice.iloc[:, -1]

            label_slice = labels_df[t_start:t_end]['label']
            if label_slice.empty: continue
            label = label_slice.mode()[0]
            
            row = {'window_id': current_sec, 'label': label}
            row.update(self.extract_bvp_features(bvp_win))
            row.update(self.extract_acc_features(acc_win))
            row.update(self.extract_eda_features(eda_win))
            row.update(self.extract_hr_features(hr_win))
            row.update(self.extract_hrv_features(ibi_win))
            
            features_list.append(row)
            
        return pd.DataFrame(features_list)