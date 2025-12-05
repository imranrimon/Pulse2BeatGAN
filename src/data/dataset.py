import torch
from torch.utils.data import Dataset
import glob
import os
import numpy as np
from .preprocessing import (
    load_and_process_bidmc_file, 
    load_and_process_pickle_file, 
    load_and_process_csv_file,
    create_peak_mask
)
from tqdm import tqdm

class BaseDataset(Dataset):
    def __init__(self, data_dir, target_sr=128, segment_length=512, cache_data=True, limit=None):
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.segment_length = segment_length
        self.ecg_segments = []
        self.ppg_segments = []
        self.cache_data = cache_data
        self.limit = limit

    def _load_data(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.ecg_segments)

    def __getitem__(self, idx):
        ecg = self.ecg_segments[idx]
        ppg = self.ppg_segments[idx]
        
        # Convert to tensor
        ecg_tensor = torch.from_numpy(ecg).float()
        ppg_tensor = torch.from_numpy(ppg).float()
        
        # Downsampling for multi-stage training
        # Target sizes: 512 (original), 256, 128
        ecg_256 = torch.nn.functional.interpolate(ecg_tensor.view(1, 1, -1), size=256, mode='linear', align_corners=False).view(-1)
        ecg_128 = torch.nn.functional.interpolate(ecg_tensor.view(1, 1, -1), size=128, mode='linear', align_corners=False).view(-1)
        
        ppg_256 = torch.nn.functional.interpolate(ppg_tensor.view(1, 1, -1), size=256, mode='linear', align_corners=False).view(-1)
        ppg_128 = torch.nn.functional.interpolate(ppg_tensor.view(1, 1, -1), size=128, mode='linear', align_corners=False).view(-1)
        
        return ppg_128, ppg_256, ppg_tensor, ecg_128, ecg_256, ecg_tensor

class BIDMCDataset(BaseDataset):
    def __init__(self, data_dir, target_sr=128, segment_length=512, cache_data=True, limit=None):
        super().__init__(data_dir, target_sr, segment_length, cache_data, limit)
        self.files = glob.glob(os.path.join(data_dir, "*_Signals.csv"))
        if self.limit:
            self.files = self.files[:self.limit]
        if self.cache_data:
            self._load_data()
            
    def _load_data(self):
        print(f"Loading BIDMC from {self.data_dir}...")
        for f in tqdm(self.files):
            ecg, ppg = load_and_process_bidmc_file(f, self.target_sr, self.segment_length)
            if len(ecg) > 0:
                self.ecg_segments.append(ecg)
                self.ppg_segments.append(ppg)
        self._finalize_data()

    def _finalize_data(self):
        if len(self.ecg_segments) > 0:
            self.ecg_segments = np.concatenate(self.ecg_segments, axis=0)
            self.ppg_segments = np.concatenate(self.ppg_segments, axis=0)
        else:
            print("Warning: No data loaded.")
            self.ecg_segments = np.array([])
            self.ppg_segments = np.array([])
        print(f"Total segments: {len(self.ecg_segments)}")

class WESADDataset(BaseDataset):
    def __init__(self, data_dir, target_sr=128, segment_length=512, cache_data=True, limit=None):
        super().__init__(data_dir, target_sr, segment_length, cache_data, limit)
        # WESAD structure: subject folders / S*.pkl
        self.files = glob.glob(os.path.join(data_dir, "**", "*.pkl"), recursive=True)
        if self.limit:
            self.files = self.files[:self.limit]
        if self.cache_data:
            self._load_data()

    def _load_data(self):
        print(f"Loading WESAD from {self.data_dir}...")
        for f in tqdm(self.files):
            ecg, ppg = load_and_process_pickle_file(f, self.target_sr, self.segment_length, dataset_type='wesad')
            if len(ecg) > 0:
                self.ecg_segments.append(ecg)
                self.ppg_segments.append(ppg)
        self._finalize_data()
        
    def _finalize_data(self):
        if len(self.ecg_segments) > 0:
            self.ecg_segments = np.concatenate(self.ecg_segments, axis=0)
            self.ppg_segments = np.concatenate(self.ppg_segments, axis=0)
        else:
            print("Warning: No data loaded.")
            self.ecg_segments = np.array([])
            self.ppg_segments = np.array([])
        print(f"Total segments: {len(self.ecg_segments)}")

class DALIADataset(BaseDataset):
    def __init__(self, data_dir, target_sr=128, segment_length=512, cache_data=True, limit=None):
        super().__init__(data_dir, target_sr, segment_length, cache_data, limit)
        self.files = glob.glob(os.path.join(data_dir, "**", "*.pkl"), recursive=True)
        if self.limit:
            self.files = self.files[:self.limit]
        if self.cache_data:
            self._load_data()

    def _load_data(self):
        print(f"Loading DALIA from {self.data_dir}...")
        for f in tqdm(self.files):
            ecg, ppg = load_and_process_pickle_file(f, self.target_sr, self.segment_length, dataset_type='dalia')
            if len(ecg) > 0:
                self.ecg_segments.append(ecg)
                self.ppg_segments.append(ppg)
        self._finalize_data()
        
    def _finalize_data(self):
        if len(self.ecg_segments) > 0:
            self.ecg_segments = np.concatenate(self.ecg_segments, axis=0)
            self.ppg_segments = np.concatenate(self.ppg_segments, axis=0)
        else:
            print("Warning: No data loaded.")
            self.ecg_segments = np.array([])
            self.ppg_segments = np.array([])
        print(f"Total segments: {len(self.ecg_segments)}")

class CapnobaseDataset(BaseDataset):
    def __init__(self, data_dir, target_sr=128, segment_length=512, cache_data=True, limit=None):
        super().__init__(data_dir, target_sr, segment_length, cache_data, limit)
        self.files = glob.glob(os.path.join(data_dir, "*.csv")) # Assuming flat directory or adjust
        if self.limit:
            self.files = self.files[:self.limit]
        if self.cache_data:
            self._load_data()

    def _load_data(self):
        print(f"Loading Capnobase from {self.data_dir}...")
        for f in tqdm(self.files):
            # Capnobase: ecg_y, pleth_y. Original SR often 300Hz or similar.
            ecg, ppg = load_and_process_csv_file(f, self.target_sr, self.segment_length, 
                                                 ecg_col='ecg_y', ppg_col='pleth_y', original_sr=None) # Let it infer or set if known
            if len(ecg) > 0:
                self.ecg_segments.append(ecg)
                self.ppg_segments.append(ppg)
        self._finalize_data()
        
    def _finalize_data(self):
        if len(self.ecg_segments) > 0:
            self.ecg_segments = np.concatenate(self.ecg_segments, axis=0)
            self.ppg_segments = np.concatenate(self.ppg_segments, axis=0)
        else:
            print("Warning: No data loaded.")
            self.ecg_segments = np.array([])
            self.ppg_segments = np.array([])
        print(f"Total segments: {len(self.ecg_segments)}")

class MIMICAFibDataset(BaseDataset):
    def __init__(self, data_dir, target_sr=128, segment_length=512, cache_data=True, limit=None):
        super().__init__(data_dir, target_sr, segment_length, cache_data, limit)
        self.files = glob.glob(os.path.join(data_dir, "**", "*.csv"), recursive=True)
        if self.limit:
            self.files = self.files[:self.limit]
        if self.cache_data:
            self._load_data()

    def _load_data(self):
        print(f"Loading MIMIC-AFib from {self.data_dir}...")
        for f in tqdm(self.files):
            # MIMIC: ECG, PPG.
            ecg, ppg = load_and_process_csv_file(f, self.target_sr, self.segment_length, 
                                                 ecg_col='ECG', ppg_col='PPG', original_sr=125) # MIMIC usually 125Hz
            if len(ecg) > 0:
                self.ecg_segments.append(ecg)
                self.ppg_segments.append(ppg)
        self._finalize_data()
        
    def _finalize_data(self):
        if len(self.ecg_segments) > 0:
            self.ecg_segments = np.concatenate(self.ecg_segments, axis=0)
            self.ppg_segments = np.concatenate(self.ppg_segments, axis=0)
        else:
            print("Warning: No data loaded.")
            self.ecg_segments = np.array([])
            self.ppg_segments = np.array([])
        print(f"Total segments: {len(self.ecg_segments)}")

class UQVitalSignsDataset(BaseDataset):
    def __init__(self, data_dir, target_sr=128, segment_length=512, cache_data=True, limit=None):
        super().__init__(data_dir, target_sr, segment_length, cache_data, limit)
        self.files = glob.glob(os.path.join(data_dir, "**", "*fulldata*.csv"), recursive=True)
        if self.limit:
            self.files = self.files[:self.limit]
        if self.cache_data:
            self._load_data()

    def _load_data(self):
        print(f"Loading UQVitalSigns from {self.data_dir}...")
        for f in tqdm(self.files):
            # UQ: ECG, Pleth.
            ecg, ppg = load_and_process_csv_file(f, self.target_sr, self.segment_length, 
                                                 ecg_col='ECG', ppg_col='Pleth', original_sr=100) # UQ often 100Hz or 125Hz
            if len(ecg) > 0:
                self.ecg_segments.append(ecg)
                self.ppg_segments.append(ppg)
        self._finalize_data()
        
    def _finalize_data(self):
        if len(self.ecg_segments) > 0:
            self.ecg_segments = np.concatenate(self.ecg_segments, axis=0)
            self.ppg_segments = np.concatenate(self.ppg_segments, axis=0)
        else:
            print("Warning: No data loaded.")
            self.ecg_segments = np.array([])
            self.ppg_segments = np.array([])
        print(f"Total segments: {len(self.ecg_segments)}")
