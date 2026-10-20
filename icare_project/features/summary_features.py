import numpy as np

def extract_classical_features(bci_sequence, obs_mask=None):
    """
    Extracts summary features from a (25920,) BCI sequence for classical ML.
    Assumes 1 point = 10 seconds (360 points/hr, 25920 points = 72 hours).
    bci_sequence: 1D numpy array of shape (25920,)
    obs_mask: optional 1D binary/boolean array (25920,) indicating observed (monitored) time points.
              When provided, suppression fractions and summary statistics are calculated strictly
              over monitored intervals rather than unrecorded zero-padded intervals.
    """
    seq = np.array(bci_sequence).squeeze()
    if obs_mask is not None:
        mask = np.array(obs_mask).squeeze().astype(bool)
        observed = seq[mask]
    else:
        mask = None
        observed = seq
    
    # Basic Stats (computed on observed segments when mask provided)
    if len(observed) > 0:
        mean_val = float(np.mean(observed))
        min_val = float(np.min(observed))
        max_val = float(np.max(observed))
        std_val = float(np.std(observed))
    else:
        mean_val, min_val, max_val, std_val = 0.0, 0.0, 0.0, 0.0
    
    # Linear slope
    if mask is not None and len(observed) > 1:
        x_obs = np.where(mask)[0]
        slope, _ = np.polyfit(x_obs, observed, 1)
    else:
        x = np.arange(len(seq))
        slope, _ = np.polyfit(x, seq, 1)
    
    # Fractions of sequence spent below continuity thresholds
    # Evaluated over monitored intervals when obs_mask is provided
    if len(observed) > 0:
        frac_lt_01 = float(np.mean(observed < 0.1))
        frac_lt_02 = float(np.mean(observed < 0.2))
        frac_lt_03 = float(np.mean(observed < 0.3))
    else:
        frac_lt_01, frac_lt_02, frac_lt_03 = 0.0, 0.0, 0.0
    
    # Time to first sustained continuity (>0.7 for N consecutive windows)
    # N = 36 corresponds to 6 minutes of sustained continuity (36 * 10s = 360s)
    # Stored as sample index (0 to 25,920 steps; 360 steps = 1 hour), standard-scaled for ML baselines
    sustained_n = 36
    is_continuous = (seq > 0.7).astype(int)
    conv = np.convolve(is_continuous, np.ones(sustained_n), mode='valid')
    idx_first_sustained = np.argmax(conv == sustained_n)
    if conv[idx_first_sustained] == sustained_n:
        time_to_continuity = float(idx_first_sustained)
    else:
        time_to_continuity = float(len(seq)) # Patient never achieved sustained continuity (max index 25920)
        
    # Rolling stats (1 min = 6 pts, 5 min = 30 pts, 30 min = 180 pts)
    features = [mean_val, min_val, max_val, std_val, slope, frac_lt_01, frac_lt_02, frac_lt_03, time_to_continuity]
    
    for window in [6, 30, 180]:
        if len(seq) < window:
            rolling_mean_max = 0.0; rolling_var_max = 0.0
            rolling_mean_min = 0.0; rolling_var_min = 0.0
        else:
            # Efficient moving average
            ret = np.cumsum(seq, dtype=float)
            ret[window:] = ret[window:] - ret[:-window]
            rolling_mean = ret[window - 1:] / window
            
            # Efficient moving variance (Var(x) = E[x^2] - E[x]^2)
            ret_sq = np.cumsum(seq**2, dtype=float)
            ret_sq[window:] = ret_sq[window:] - ret_sq[:-window]
            rolling_mean_sq = ret_sq[window - 1:] / window
            
            rolling_var = np.maximum(rolling_mean_sq - rolling_mean**2, 0.0)
            
            rolling_mean_max = float(np.max(rolling_mean))
            rolling_mean_min = float(np.min(rolling_mean))
            rolling_var_max = float(np.max(rolling_var))
            rolling_var_min = float(np.min(rolling_var))
            
        features.extend([rolling_mean_max, rolling_mean_min, rolling_var_max, rolling_var_min])
        
    # Return 9 + 3*4 = 21 features
    return np.array(features)

def build_baseline_dataset(bci_batch, meta_batch, mask_batch=None):
    """
    bci_batch: (Batch, 25920, 1) or similar iterable
    meta_batch: (Batch, 6)
    mask_batch: optional (Batch, 25920) observation masks
    Returns flat (Batch, 21 + 6)
    """
    X = []
    for i in range(len(bci_batch)):
        mask = mask_batch[i] if mask_batch is not None else None
        bci_feats = extract_classical_features(bci_batch[i], obs_mask=mask)
        meta_feats = np.array(meta_batch[i])
        X.append(np.concatenate([bci_feats, meta_feats]))
    return np.array(X)
