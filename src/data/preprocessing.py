import numpy as np
import pandas as pd
import neurokit2 as nk
import librosa
import sklearn.preprocessing as skp
from scipy import signal
import pickle

def filter_ecg(signal_data, sampling_rate):
    signal_data = np.array(signal_data)
    # Using neurokit for robust cleaning
    try:
        return nk.ecg_clean(signal_data, sampling_rate=sampling_rate, method="neurokit")
    except Exception as e:
        print(f"Error cleaning ECG: {e}")
        return signal_data

def filter_ppg(signal_data, sampling_rate):
    try:
        return nk.ppg_clean(signal_data, sampling_rate=sampling_rate, method="elgendi")
    except Exception as e:
        print(f"Error cleaning PPG: {e}")
        return signal_data

def normalize_and_segment(ecg, ppg, segment_length=512):
    # Normalize (Z-score)
    ecg = (ecg - np.mean(ecg)) / (np.std(ecg) + 1e-6)
    ppg = (ppg - np.mean(ppg)) / (np.std(ppg) + 1e-6)
    
    segments_ecg = []
    segments_ppg = []
    
    overlap = int(segment_length * 0.1)
    step = segment_length - overlap
    
    for i in range(0, len(ecg) - segment_length, step):
        seg_ecg = ecg[i:i+segment_length]
        seg_ppg = ppg[i:i+segment_length]
        
        # MinMax Scale per segment to (-1, 1)
        scaler = skp.MinMaxScaler(feature_range=(-1, 1))
        seg_ecg = scaler.fit_transform(seg_ecg.reshape(-1, 1)).flatten()
        seg_ppg = scaler.fit_transform(seg_ppg.reshape(-1, 1)).flatten()
        
        segments_ecg.append(seg_ecg)
        segments_ppg.append(seg_ppg)
        
    return np.array(segments_ecg), np.array(segments_ppg)

def load_and_process_bidmc_file(file_path, target_sr=128, segment_length=512):
    try:
        df = pd.read_csv(file_path)
        if ' II' in df.columns: ecg = df[' II'].values
        elif 'II' in df.columns: ecg = df['II'].values
        else: ecg = df.iloc[:, 1].values
            
        if ' PLETH' in df.columns: ppg = df[' PLETH'].values
        elif 'PLETH' in df.columns: ppg = df['PLETH'].values
        else: ppg = df.iloc[:, 2].values

        original_sr = 125
        
        if original_sr != target_sr:
            ecg = librosa.resample(ecg, orig_sr=original_sr, target_sr=target_sr)
            ppg = librosa.resample(ppg, orig_sr=original_sr, target_sr=target_sr)
            
        ecg = filter_ecg(ecg, target_sr)
        ppg = filter_ppg(ppg, target_sr)
        
        return normalize_and_segment(ecg, ppg, segment_length)
    except Exception as e:
        print(f"Error processing BIDMC {file_path}: {e}")
        return np.array([]), np.array([])

def load_and_process_pickle_file(file_path, target_sr=128, segment_length=512, dataset_type='wesad'):
    """
    Loads WESAD or DALIA pickle files.
    Structure assumed: data['signal']['chest']['ECG'] and data['signal']['wrist']['BVP']
    """
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            
        # Extract signals based on dataset structure
        # WESAD and DALIA often have similar structures for the aligned data
        if 'signal' in data:
            ecg = data['signal']['chest']['ECG'].flatten()
            ppg = data['signal']['wrist']['BVP'].flatten()
            
            # Sampling rates
            # WESAD: ECG 700Hz, BVP 64Hz
            # DALIA: ECG 700Hz, BVP 32Hz (need to verify, assuming standard)
            # We will resample everything to target_sr
            
            ecg_sr = 700
            ppg_sr = 64 if dataset_type == 'wesad' else 32 # DALIA BVP is often 32Hz or 64Hz, need to be careful. 
            # Let's assume 64 for WESAD. For DALIA, the paper says 32Hz for PPG (E4 wristband).
            if dataset_type == 'dalia':
                ppg_sr = 32
                
            if ecg_sr != target_sr:
                ecg = librosa.resample(ecg, orig_sr=ecg_sr, target_sr=target_sr)
            if ppg_sr != target_sr:
                ppg = librosa.resample(ppg, orig_sr=ppg_sr, target_sr=target_sr)
                
            # Ensure lengths match after resampling (might differ slightly due to rounding)
            min_len = min(len(ecg), len(ppg))
            ecg = ecg[:min_len]
            ppg = ppg[:min_len]
            
            ecg = filter_ecg(ecg, target_sr)
            ppg = filter_ppg(ppg, target_sr)
            
            return normalize_and_segment(ecg, ppg, segment_length)
            
        else:
            print(f"Key 'signal' not found in {file_path}")
            return np.array([]), np.array([])

    except Exception as e:
        print(f"Error processing Pickle {file_path}: {e}")
        return np.array([]), np.array([])

def load_and_process_csv_file(file_path, target_sr=128, segment_length=512, ecg_col='ECG', ppg_col='PPG', original_sr=None):
    """
    Generic CSV loader.
    """
    try:
        try:
            df = pd.read_csv(file_path, on_bad_lines='skip')
        except TypeError:
             # Fallback for older pandas
            df = pd.read_csv(file_path, error_bad_lines=False)
        
        # Handle column finding with some flexibility
        if ecg_col not in df.columns:
            # Try finding a column that contains the name
            candidates = [c for c in df.columns if ecg_col.lower() in c.lower()]
            if candidates:
                ecg_col = candidates[0]
            else:
                # Fallback to index if needed, but better to fail safely
                pass
                
        if ppg_col not in df.columns:
            candidates = [c for c in df.columns if ppg_col.lower() in c.lower()]
            if candidates:
                ppg_col = candidates[0]
        
        if ecg_col in df.columns and ppg_col in df.columns:
            ecg = df[ecg_col].values
            ppg = df[ppg_col].values
            
            # Handle NaN
            if np.isnan(ecg).any():
                df = df.dropna(subset=[ecg_col, ppg_col])
                ecg = df[ecg_col].values
                ppg = df[ppg_col].values

            # Determine SR if not provided?
            # For now, require original_sr or infer from Time column if exists
            if original_sr is None:
                if 'Time' in df.columns:
                    # Infer from first few samples
                    try:
                        t = df['Time'].values
                        if isinstance(t[0], str):
                             # Handle string time? Skip for now, assume fixed SR if passed
                             pass
                        else:
                            diffs = np.diff(t[:100])
                            original_sr = int(1 / np.mean(diffs))
                    except:
                        original_sr = 128 # Fallback
                else:
                    original_sr = 128 # Fallback assumption
            
            if original_sr != target_sr:
                ecg = librosa.resample(ecg, orig_sr=original_sr, target_sr=target_sr)
                ppg = librosa.resample(ppg, orig_sr=original_sr, target_sr=target_sr)
                
            ecg = filter_ecg(ecg, target_sr)
            ppg = filter_ppg(ppg, target_sr)
            
            return normalize_and_segment(ecg, ppg, segment_length)
        else:
            print(f"Columns {ecg_col} or {ppg_col} not found in {file_path}. Columns: {df.columns}")
            return np.array([]), np.array([])

    except Exception as e:
        print(f"Error processing CSV {file_path}: {e}")
        return np.array([]), np.array([])

def create_peak_mask(signal_data, sampling_rate=128, return_indices=False):
    # Detect peaks
    # R-peaks for ECG
    try:
        _, rpeaks = nk.ecg_peaks(signal_data, sampling_rate=sampling_rate)
        rpeaks_indices = rpeaks['ECG_R_Peaks']
        
        # Create mask
        mask = np.zeros_like(signal_data)
        window = int(0.1 * sampling_rate) 
        
        for peak in rpeaks_indices:
            if np.isnan(peak): continue
            start = max(0, int(peak - window))
            end = min(len(signal_data), int(peak + window))
            mask[start:end] = 1.0
            
        if return_indices:
            return mask, rpeaks_indices
        return mask
    except:
        if return_indices:
            return np.zeros_like(signal_data), np.array([])
        return np.zeros_like(signal_data)
