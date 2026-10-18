import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model/cnn_bigru_model')))
from data_loader import ICAREDataset

# ==============================================================================
# Leave-One-Hospital-Out (LOHO) Data Splitting
# ==============================================================================
# In each of the 5 folds, one entire hospital is kept completely secret for testing.
# The remaining 4 hospitals are split 80% for training and 20% for validation.
# ==============================================================================

HOSPITAL_MAP = ['A', 'B', 'D', 'E', 'F']

def get_unified_loaders(dataset, fold_idx, batch_size=16, val_split=0.2, seed=42):
    """
    Creates PyTorch data loaders for:
    1. Training set (from 4 development hospitals)
    2. Validation set (from 4 development hospitals, used for tuning and calibration)
    3. Test set (from the 1 held-out hospital, evaluated only once)
    """
    if fold_idx < 0 or fold_idx >= len(HOSPITAL_MAP):
        raise ValueError(f"Fold index must be 0 to 4 (found {fold_idx})")
        
    leave_out_hospital = HOSPITAL_MAP[fold_idx]
    hospitals = dataset.hospitals
    
    # Separate the held-out test hospital from the rest
    test_indices = [i for i, h in enumerate(hospitals) if h == leave_out_hospital]
    remaining_indices = [i for i, h in enumerate(hospitals) if h != leave_out_hospital]
    
    # Split the remaining 4 hospitals into 80% train and 20% validation
    np.random.seed(seed)
    np.random.shuffle(remaining_indices)
    
    split = int(np.floor(val_split * len(remaining_indices)))
    val_indices = remaining_indices[:split]
    train_indices = remaining_indices[split:]
    
    # Balanced sampler for training
    train_sex_subset = [dataset.metadata[i, 1] for i in train_indices]
    count_0 = sum(1 for s in train_sex_subset if s == 0) # Female count
    count_1 = sum(1 for s in train_sex_subset if s == 1) # Male count
    
    w0 = 1.0 / count_0 if count_0 > 0 else 0
    w1 = 1.0 / count_1 if count_1 > 0 else 0
    
    subset_weights = [w1 if s == 1 else w0 for s in train_sex_subset]
    subset_weights = torch.tensor(subset_weights, dtype=torch.double)
    sampler = WeightedRandomSampler(subset_weights, len(subset_weights))
    
    # Build standard PyTorch subsets
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    test_subset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, leave_out_hospital
