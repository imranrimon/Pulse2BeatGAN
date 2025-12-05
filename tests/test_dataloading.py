import sys
import os
import torch
from torch.utils.data import DataLoader

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset import (
    BIDMCDataset, 
    DALIADataset, 
    WESADDataset, 
    CapnobaseDataset, 
    MIMICAFibDataset, 
    UQVitalSignsDataset
)

def test_dataset(dataset_name, dataset_class, data_dir):
    print(f"\nTesting {dataset_name} from {data_dir}...")
    try:
        # Initialize dataset with cache_data=True to trigger loading, limit to 1 file
        dataset = dataset_class(data_dir, target_sr=128, segment_length=512, cache_data=True, limit=1)
        
        if len(dataset) == 0:
            print(f"Warning: {dataset_name} is empty.")
            return
            
        print(f"Loaded {len(dataset)} segments.")
        
        # Test __getitem__
        item = dataset[0]
        # Expect tuple of 6 tensors
        if len(item) != 6:
            print(f"Error: Expected 6 items, got {len(item)}")
            return
            
        ppg_128, ppg_256, ppg, ecg_128, ecg_256, ecg = item
        
        print(f"Shapes: PPG_128: {ppg_128.shape}, PPG_256: {ppg_256.shape}, PPG: {ppg.shape}")
        print(f"Shapes: ECG_128: {ecg_128.shape}, ECG_256: {ecg_256.shape}, ECG: {ecg.shape}")
        
        # Check shapes
        assert ppg.shape == (512,)
        assert ecg.shape == (512,)
        assert ppg_128.shape == (128,)
        assert ecg_128.shape == (128,)
        
        print(f"{dataset_name} Passed!")
        
    except Exception as e:
        print(f"Error testing {dataset_name}: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    base_dir = "d:/Pulse2BeatGAN/Dataset"
    
    datasets = [
        ("BIDMC", BIDMCDataset, os.path.join(base_dir, "BIDMC")),
        ("Capnobase", CapnobaseDataset, os.path.join(base_dir, "Capnobase")),
        ("DaLia", DALIADataset, os.path.join(base_dir, "DaLia")),
        ("MIMIC-AFib", MIMICAFibDataset, os.path.join(base_dir, "MIMIC-AFib")),
        ("WESAD", WESADDataset, os.path.join(base_dir, "WESAD")),
        ("uqvitalsignsdata", UQVitalSignsDataset, os.path.join(base_dir, "uqvitalsignsdata")),
    ]
    
    for name, cls, path in datasets:
        test_dataset(name, cls, path)
