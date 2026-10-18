import numpy as np

# ==============================================================================
# Engineered Features for Classical Machine Learning
# ==============================================================================
# Traditional models (XGBoost, Random Forest, Logistic Regression) cannot
# easily read 25,920 raw time steps directly.
# This function calculates 21 simple statistical summaries from each patient's
# 72-hour EEG continuity recording.
# ==============================================================================

def extract_classical_features(bci_sequence):
    """
    Computes 21 summary features from a single patient's 72-hour BCI sequence:
    - 4 basic stats: overall mean, min, max, standard deviation
    - 1 trend feature: linear slope over 72 hours
    - 3 suppression features: fraction of time spent suppressed below 0.1, 0.2, and 0.3
    - 1 recovery milestone: time until first 6 minutes of sustained continuity (> 0.7)
    - 12 rolling window features: rolling mean & variance over 1-min, 5-min, and 30-min windows
    """
    seq = np.array(bci_sequence).squeeze()
    
    # 1. Basic Stats
    mean_val = float(np.mean(seq))
    min_val = float(np.min(seq))
    max_val = float(np.max(seq))
    std_val = float(np.std(seq))
    
    # 2. Trend (slope over 72 hours)
    x = np.arange(len(seq))
    slope, _ = np.polyfit(x, seq, 1)
    
    # 3. Suppression burden (fraction of time EEG was deeply suppressed)
    frac_lt_01 = float(np.mean(seq < 0.1))
    frac_lt_02 = float(np.mean(seq < 0.2))
    frac_lt_03 = float(np.mean(seq < 0.3))
    
    # 4. Time to first sustained continuity (> 0.7 for 36 consecutive steps = 6 minutes)
    sustained_n = 36
    is_continuous = (seq > 0.7).astype(int)
    conv = np.convolve(is_continuous, np.ones(sustained_n), mode='valid')
    idx_first = np.argmax(conv == sustained_n)
    if conv[idx_first] == sustained_n:
        time_to_continuity = float(idx_first)
    else:
        time_to_continuity = float(len(seq)) # Patient never achieved sustained continuity
        
    features = [mean_val, min_val, max_val, std_val, slope, frac_lt_01, frac_lt_02, frac_lt_03, time_to_continuity]
    
    # 5. Short, medium, and long rolling windows (1 min = 6 points, 5 min = 30, 30 min = 180)
    for window in [6, 30, 180]:
        if len(seq) < window:
            features.extend([0.0, 0.0, 0.0, 0.0])
        else:
            # Moving average
            ret = np.cumsum(seq, dtype=float)
            ret[window:] = ret[window:] - ret[:-window]
            rolling_mean = ret[window - 1:] / window
            
            # Moving variance
            ret_sq = np.cumsum(seq**2, dtype=float)
            ret_sq[window:] = ret_sq[window:] - ret_sq[:-window]
            rolling_mean_sq = ret_sq[window - 1:] / window
            rolling_var = np.maximum(rolling_mean_sq - rolling_mean**2, 0.0)
            
            features.extend([
                float(np.max(rolling_mean)),
                float(np.min(rolling_mean)),
                float(np.max(rolling_var)),
                float(np.min(rolling_var))
            ])
            
    return np.array(features, dtype=np.float32)
