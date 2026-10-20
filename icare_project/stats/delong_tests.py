import numpy as np
import scipy.stats

def _delong_placements(y_true, scores):
    """
    Computes structural components (placements) V10 and V01 for DeLong's test
    following the standard U-statistic formulation (DeLong et al. 1988, Sun & Xu 2014).
    
    y_true: 1D array of binary ground truth (0 or 1)
    scores: 2D array of shape (N, K) containing scores for K models
    """
    y_true = np.asarray(y_true).ravel()
    scores = np.asarray(scores)
    if scores.ndim == 1:
        scores = scores[:, np.newaxis]
        
    case_mask = (y_true == 1)
    ctrl_mask = (y_true == 0)
    
    m = np.sum(case_mask) # Number of positive cases
    n = np.sum(ctrl_mask) # Number of controls
    
    if m == 0 or n == 0:
        raise ValueError("Ground truth must contain at least one positive and one negative sample.")
        
    k = scores.shape[1]
    V10 = np.zeros((m, k), dtype=np.float64)
    V01 = np.zeros((n, k), dtype=np.float64)
    
    for col in range(k):
        s_case = scores[case_mask, col]
        s_ctrl = scores[ctrl_mask, col]
        
        # Matrix of comparisons: shape (m, n)
        # psi(X_i, Y_j) = 1 if X_i > Y_j, 0.5 if X_i == Y_j, 0 if X_i < Y_j
        diff = s_case[:, np.newaxis] - s_ctrl[np.newaxis, :]
        psi = np.where(diff > 0, 1.0, np.where(diff == 0, 0.5, 0.0))
        
        V10[:, col] = np.mean(psi, axis=1) # average over controls
        V01[:, col] = 1.0 - np.mean(psi, axis=0) # average over cases for P(Y_j > X_i)
        
    theta = np.mean(V10, axis=0) # AUROC for each model
    return V10, V01, theta, m, n

def delong_covariance(y_true, scores):
    """
    Calculates the exact DeLong covariance matrix for K models.
    """
    V10, V01, theta, m, n = _delong_placements(y_true, scores)
    
    # Sample covariance matrices with Bessel's correction
    S10 = np.cov(V10, rowvar=False) if m > 1 else np.zeros((scores.shape[1], scores.shape[1]))
    S01 = np.cov(V01, rowvar=False) if n > 1 else np.zeros((scores.shape[1], scores.shape[1]))
    
    S = S10 / m + S01 / n
    return np.atleast_2d(S), theta

def delong_roc_test(ground_truth, predictions_one, predictions_two):
    """
    Computes two-sided p-value and z-score comparing two ROC curves via DeLong's test.
    """
    y_true = np.asarray(ground_truth).ravel()
    p1 = np.asarray(predictions_one).ravel()
    p2 = np.asarray(predictions_two).ravel()
    
    scores = np.column_stack([p1, p2])
    S, theta = delong_covariance(y_true, scores)
    
    diff = theta[0] - theta[1]
    variance = S[0, 0] + S[1, 1] - 2.0 * S[0, 1]
    
    if variance <= 1e-12:
        return 1.0, 0.0
        
    z = diff / np.sqrt(variance)
    p_value = 2.0 * (1.0 - scipy.stats.norm.cdf(np.abs(z)))
    return float(np.clip(p_value, 0.0, 1.0)), float(z)

if __name__ == "__main__":
    y = np.array([1, 0, 1, 0, 1, 0, 0, 1])
    s1 = np.array([0.9, 0.1, 0.8, 0.2, 0.7, 0.3, 0.4, 0.6])
    s2 = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    p, z = delong_roc_test(y, s1, s2)
    print(f"DeLong test: p-val={p:.4f}, z={z:.4f}")
