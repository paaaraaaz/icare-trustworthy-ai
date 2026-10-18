import numpy as np
from sklearn.metrics import f1_score

# ==============================================================================
# Paired Bootstrap Test for F1-score
# ==============================================================================
# Randomly samples patients with replacement 1,000 times to calculate
# whether one model's F1-score is reliably better than another.
# ==============================================================================

def paired_bootstrap_test_f1(y_true, y_pred1, y_pred2, n_bootstraps=1000, seed=42):
    """
    Returns the p-value comparing F1 scores between Model 1 and Model 2.
    """
    np.random.seed(seed)
    n = len(y_true)
    count = 0
    
    for _ in range(n_bootstraps):
        idx = np.random.choice(n, size=n, replace=True)
        if len(np.unique(y_true[idx])) < 2:
            continue
            
        f1_1 = f1_score(y_true[idx], y_pred1[idx], zero_division=0)
        f1_2 = f1_score(y_true[idx], y_pred2[idx], zero_division=0)
        
        # Count how often model 1 is better than or equal to model 2
        if f1_1 >= f1_2:
            count += 1
            
    p_value = 1.0 - (count / n_bootstraps)
    # Two-sided empirical p-value
    p_value = 2.0 * min(p_value, 1.0 - p_value)
    return float(np.clip(p_value, 0.0, 1.0))
