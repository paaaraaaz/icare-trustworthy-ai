
import numpy as np
import scipy.io
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import os
import glob

class ICAREDataset(Dataset):
    def __init__(self, mat_file_path, metadata_root='data/physionet_metadata/training', window_hours=72, resolution_seconds=10):
        """
        Args:
            mat_file_path: path to the .mat file containing BCI features.
            metadata_root: root directory of the downloaded metadata (e.g., data/physionet_metadata/training)
            window_hours: length of the time window to consider (default 72h).
            resolution_seconds: esolution of BCI data (default is 10s).
        """
        self.window_hours = window_hours
        self.resolution_seconds = resolution_seconds
        self.total_steps = int(window_hours * 3600 / resolution_seconds)
        self.metadata_root = metadata_root
        
        # Load Data
        print(f"Loading {mat_file_path}...")
        try:
            self.raw_data = scipy.io.loadmat(mat_file_path)['ICARE_BCI_Train']
        except NotImplementedError:
             raise Exception("Please implement h5py loading if scipy.io fails (v7.3 mat files)")

        # Process Data and Load Real Metadata
        self.subjects, self.X, self.metadata, self.y, self.hospitals = self._process_data()
        
        # Validate data
        if len(self.subjects) == 0:
            print("WARNING: No subjects loaded. Check paths and data files.")

    def _process_data(self):
        # 1. Parse the extracted BCI feature data and group time-series sequences by unique patient IDs.
        
        def clean_sub_id(val):
            if hasattr(val, 'tolist'): val = val.tolist()
            if isinstance(val, list): val = val[0] if len(val) > 0 else ""
            if isinstance(val, np.ndarray): val = val.flat[0] if val.size > 0 else ""
            return str(val).strip()

        # Extract unique subject prefixes from .mat file
        raw_subjects = [clean_sub_id(x[0]) for x in self.raw_data]
        # Filter for valid IDs (ICARE_xxxx)
        patient_ids = sorted(list(set([s.split('_')[0] + '_' + s.split('_')[1] for s in raw_subjects if '_' in s])))
        
        print(f"Found {len(patient_ids)} unique patients in .mat file.")
        
        # Pre-group BCI data by patient
        patient_data_map = {}
        for row in self.raw_data:
            sub_id_full = clean_sub_id(row[0])
            parts = sub_id_full.split('_')
            if len(parts) >= 2:
                pid = f"{parts[0]}_{parts[1]}" # e.g. ICARE_0284
            else:
                continue

            if pid not in patient_data_map:
                patient_data_map[pid] = []
            
            time_val = row[1].item()
            bci_chunk = row[2]
            if bci_chunk.ndim > 1: bci_chunk = bci_chunk.flatten()
            patient_data_map[pid].append((time_val, bci_chunk))

        # 2.Build Dataset with Real Metadata
        X_list = []
        meta_list = []
        y_list = []
        subj_list = []
        hospital_list = []

        
        #Hospital encoding map (letters)
        

        for pid in patient_ids:
            if pid not in patient_data_map:
                continue
            
            # Extract numeric ID for metadata file lookup
            # ICARE_0284 -> 0284
            try:
                numeric_id = pid.split('_')[1]
            except IndexError:
                continue
                
            # Find metadata file
            # Path: data/physionet_metadata/training/<ID>/<ID>.txt
            meta_path = os.path.join(self.metadata_root, numeric_id, f"{numeric_id}.txt")
            
            if not os.path.exists(meta_path):
                # print(f"Metadata not found for {pid} at {meta_path}")
                continue # Skip if no metadata
                
            # Parse Metadata
            meta_dict = self._parse_metadata_file(meta_path)
            if meta_dict is None:
                continue
                
            # Construct the structured clinical feature vector using standard continuous normalization.
            # Included Features: Age, Sex, Return of Spontaneous Circulation (ROSC), Out-of-Hospital Cardiac Arrest (OHCA), Shockable Rhythm, Targeted Temperature Management (TTM)
            # Sex: Female=0, Male=1
            sex_val = 1.0 if meta_dict.get('Sex') == 'Male' else 0.0
            
            # Age
            age_val = float(meta_dict.get('Age', 60.0)) / 100.0 # Normalize
            
            # ROSC
            # Can be NaN. Handle it.
            rosc_raw = meta_dict.get('ROSC')
            if rosc_raw is None or np.isnan(rosc_raw):
                rosc_val = 0.5 # Mean imputation (normalized approximate) or specific indicator
            else:
                rosc_val = float(rosc_raw) / 100.0 # Normalize (cap at reasonable val?)
                
            # OHCA: True/False
            ohca_val = 1.0 if meta_dict.get('OHCA') == True else 0.0
            
            # Shockable Rhythm: True/False
            shock_val = 1.0 if meta_dict.get('Shockable Rhythm') == True else 0.0
            
            # TTM: 33, 36, NaN
            ttm_raw = meta_dict.get('TTM')
            if ttm_raw == 33 or ttm_raw == 33.0:
                ttm_val = 0.33 
            elif ttm_raw == 36 or ttm_raw == 36.0:
                ttm_val = 0.36
            else:
                ttm_val = 0.0 # No TTM or NaN
                
            # Meta Vector (Size 6)
            # [Age, Sex, ROSC, OHCA, Shockable, TTM]
            meta_vec = [age_val, sex_val, rosc_val, ohca_val, shock_val, ttm_val]
            
            # Final check for NaNs in vector
            if np.any(np.isnan(meta_vec)):
                # Impute any remaining NaNs with 0
                meta_vec = [0.0 if np.isnan(v) else v for v in meta_vec]
            
            # Target: Outcome / CPC
            # Good (CPC 1-2) -> 0
            # Poor (CPC 3-5) -> 1
            cpc = meta_dict.get('CPC')
            if cpc is None:
                continue # Cannot train without target
                
            label = 1.0 if cpc >= 3 else 0.0
            
            # Hospital
            hospital = meta_dict.get('Hospital', 'Unknown')
            
            # Construct BCI Time Series (Same as before)
            segments = patient_data_map[pid]
            full_series = np.zeros(self.total_steps, dtype=np.float32)
            segments.sort(key=lambda x: x[0])
            
            for start_time, bci_values in segments:
                start_idx = int(start_time / self.resolution_seconds)
                length = len(bci_values)
                if start_idx >= self.total_steps: continue
                end_idx = start_idx + length
                if end_idx > self.total_steps:
                    write_len = self.total_steps - start_idx
                    full_series[start_idx:] = bci_values[:write_len]
                else:
                    full_series[start_idx:end_idx] = bci_values
            
            # Add to lists
            X_list.append(full_series)
            meta_list.append(meta_vec)
            y_list.append(label)
            subj_list.append(pid)
            hospital_list.append(hospital)

        return subj_list, np.array(X_list), np.array(meta_list), np.array(y_list), np.array(hospital_list)

    def _parse_metadata_file(self, path):
        # Format: "Key: Value"
        data = {}
        try:
            with open(path, 'r') as f:
                for line in f:
                    if ':' not in line: continue
                    key, val = line.split(':', 1)
                    key = key.strip()
                    val = val.strip()
                    
                    # Type conversion
                    if val == 'nan' or val == 'NaN':
                        val = np.nan
                    elif val == 'True':
                        val = True
                    elif val == 'False':
                        val = False
                    else:
                        # Try number
                        try:
                            val = float(val)
                            if val.is_integer(): val = int(val)
                        except:
                            pass # Keep as string
                    
                    data[key] = val
                    
            return data
        except Exception as e:
            print(f"Error parsing {path}: {e}")
            return None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x_bci = torch.tensor(self.X[idx], dtype=torch.float32).unsqueeze(-1)
        x_meta = torch.tensor(self.metadata[idx], dtype=torch.float32)
        y = torch.tensor(self.y[idx], dtype=torch.float32)
        return x_bci, x_meta, y

def get_loocv_loaders(dataset, leave_out_hospital, batch_size=32):
    # Filter indices based on hospital
    hospitals = dataset.hospitals
    
    train_indices = [i for i, h in enumerate(hospitals) if h != leave_out_hospital]
    val_indices = [i for i, h in enumerate(hospitals) if h == leave_out_hospital]
    
    if len(val_indices) == 0:
        print(f"Warning: No subjects found for hospital {leave_out_hospital}")
        return None, None
        
    # Training Sampler (Balanced)
    train_targets = dataset.y[train_indices]
    
    # Prioritizing biological sex balance during the sampling phase to mitigate demographic bias.
    # The expected structured metadata index for Sex is 1 (where 0=Female, 1=Male).
    train_sex = dataset.metadata[train_indices, 1]
    
    count_0 = np.sum(train_sex == 0) # Female
    count_1 = np.sum(train_sex == 1) # Male
    
    weight_0 = 1.0 / count_0 if count_0 > 0 else 0
    weight_1 = 1.0 / count_1 if count_1 > 0 else 0
    
    sample_weights = np.array([weight_1 if s == 1 else weight_0 for s in train_sex])
    sample_weights = torch.from_numpy(sample_weights).double()
    

    # Creating the Loaders
    """
    When utilizing the WeightedRandomSampler alongside the Subset module, it is necessary to provide weights 
    that strictly correspond to the localized subset indices rather than the global indices. 
    The sampler will subsequently generate indices mapped to this specific cohort.
    """

    # Correct weights calculation for the subset:
    # 1. Extract targets/attributes for the subset
    train_sex_subset = [dataset.metadata[i, 1] for i in train_indices]
    
    count_0 = sum(1 for s in train_sex_subset if s == 0)
    count_1 = sum(1 for s in train_sex_subset if s == 1)
    
    w0 = 1.0 / count_0 if count_0 > 0 else 0
    w1 = 1.0 / count_1 if count_1 > 0 else 0
    
    subset_weights = [w1 if s == 1 else w0 for s in train_sex_subset]
    subset_weights = torch.tensor(subset_weights, dtype=torch.double)
    
    # Sampler operates on the subset range
    sampler = WeightedRandomSampler(subset_weights, len(subset_weights))
    
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def get_loocv_loaders_tri_split(dataset, leave_out_hospital, batch_size=32, val_split=0.2, seed=42):
    """
    Returns (train_loader, val_loader, test_loader)
    Test: leave_out_hospital
    Train/Val: Split of remaining hospitals (1-val_split / val_split)
    """
    # Filter indices based on hospital
    hospitals = dataset.hospitals
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    
    test_indices = [i for i, h in enumerate(hospitals) if h == leave_out_hospital]
    remaining_indices = [i for i, h in enumerate(hospitals) if h != leave_out_hospital]
    
    if len(test_indices) == 0:
        print(f"Warning: No subjects found for hospital {leave_out_hospital}")
        return None, None, None
        
    # Split remaining into Train/Val
    # using random shuffle with seed for reproducibility
    np.random.seed(seed)
    np.random.shuffle(remaining_indices)
    
    split = int(np.floor(val_split * len(remaining_indices)))
    val_indices = remaining_indices[:split]
    train_indices = remaining_indices[split:]
    
    # Weighted Sampler for Training 

    # Calculate weights for Sex balance in the TRAIN set
    train_sex_subset = [dataset.metadata[i, 1] for i in train_indices]
    
    count_0 = sum(1 for s in train_sex_subset if s == 0) # Female
    count_1 = sum(1 for s in train_sex_subset if s == 1) # Male
    
    w0 = 1.0 / count_0 if count_0 > 0 else 0
    w1 = 1.0 / count_1 if count_1 > 0 else 0
    
    subset_weights = [w1 if s == 1 else w0 for s in train_sex_subset]
    subset_weights = torch.tensor(subset_weights, dtype=torch.double)
    
    sampler = WeightedRandomSampler(subset_weights, len(subset_weights))
    
    # Create Loaders
    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)
    test_subset = torch.utils.data.Subset(dataset, test_indices)
    
    train_loader = DataLoader(train_subset, batch_size=batch_size, sampler=sampler)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    # Test
    path = 'data/BCI_Features_for_all_train_ICARE_Subjects_.mat'
    ds = ICAREDataset(path)
    print(f"Total Subjects: {len(ds)}")
    if len(ds) > 0:
        print(f"Hospitals found: {np.unique(ds.hospitals)}")
        print(f"First subject meta: {ds.metadata[0]}") # [Age, Sex, ROSC, OHCA, Shock, TTM]
