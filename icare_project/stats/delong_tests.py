import numpy as np
import scipy.stats

# ==============================================================================
# DeLong Test for ROC Curves
# ==============================================================================
# This statistical test checks whether the difference in AUROC between two
# models tested on the exact same patients is statistically significant (p < 0.05).
# ==============================================================================

def compute_midrank(x):
    """Computes tied ranks for continuous predictions."""
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def calculate_delong_cov(predictions, ground_truth):
    idx1 = ground_truth == 1
    idx0 = ground_truth == 0
    m = np.sum(idx1)
    n = np.sum(idx0)
    
    V10 = np.zeros((predictions.shape[1], m))
    V01 = np.zeros((predictions.shape[1], n))
    
    for i in range(predictions.shape[1]):
        r = compute_midrank(predictions[:, i])
        V10[i, :] = r[idx1]
        V01[i, :] = r[idx0]
        
    V10 = V10 / n
    V01 = V01 / m
    theta = np.mean(V10, axis=1)
    
    S10 = np.cov(V10)
    S01 = np.cov(V01)
    S = S10 / m + S01 / n
    return S, theta

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Compares Model A vs Model B:
    Returns the p-value and the difference in AUC.
    """
    predictions = np.column_stack((predictions_one, predictions_two))
    ground_truth = np.array(ground_truth)
    
    S, theta = calculate_delong_cov(predictions, ground_truth)
    diff = theta[0] - theta[1]
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    
    if var <= 0:
        return 1.0, 0.0
        
    z = diff / np.sqrt(var)
    p_value = 2 * (1 - scipy.stats.norm.cdf(np.abs(z)))
    return float(p_value), float(diff)
