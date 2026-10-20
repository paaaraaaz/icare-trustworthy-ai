import numpy as np
from sklearn.metrics import f1_score

def paired_bootstrap_test_f1(y_true, y_pred1, y_pred2, n_bootstraps=2000, random_state=42):
    """
    Performs a paired permutation test (label-swapping under H0: no performance difference)
    to compare F1 scores between two models on the same cohort.
    
    Parameters:
    y_true (np.array): Ground truth labels.
    y_pred1 (np.array): Binary predictions from model 1.
    y_pred2 (np.array): Binary predictions from model 2.
    n_bootstraps (int): Number of permutations / bootstrap resamples.
    random_state (int): Seed for reproducibility.
    
    Returns:
    p_value (float): Two-sided permutation p-value under H0.
    ci_delta (tuple): 95% bootstrap confidence interval of F1 difference (model1 - model2).
    """
    y_true = np.asarray(y_true).ravel()
    y_pred1 = np.asarray(y_pred1).ravel()
    y_pred2 = np.asarray(y_pred2).ravel()
    
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    
    obs_f1_1 = f1_score(y_true, y_pred1, zero_division=0)
    obs_f1_2 = f1_score(y_true, y_pred2, zero_division=0)
    obs_diff = obs_f1_1 - obs_f1_2
    
    # 1. 95% Bootstrap CI of the difference
    boot_diffs = np.empty(n_bootstraps, dtype=np.float64)
    for i in range(n_bootstraps):
        idx = rng.randint(0, n, size=n)
        b_f1_1 = f1_score(y_true[idx], y_pred1[idx], zero_division=0)
        b_f1_2 = f1_score(y_true[idx], y_pred2[idx], zero_division=0)
        boot_diffs[i] = b_f1_1 - b_f1_2
    ci_lower = float(np.percentile(boot_diffs, 2.5))
    ci_upper = float(np.percentile(boot_diffs, 97.5))
    ci_delta = (ci_lower, ci_upper)
    
    # 2. Paired Permutation Test under H0 (random swap of predictions per subject)
    perm_diffs = np.empty(n_bootstraps, dtype=np.float64)
    for i in range(n_bootstraps):
        swap_mask = rng.rand(n) < 0.5
        p1_perm = np.where(swap_mask, y_pred2, y_pred1)
        p2_perm = np.where(swap_mask, y_pred1, y_pred2)
        perm_diffs[i] = f1_score(y_true, p1_perm, zero_division=0) - f1_score(y_true, p2_perm, zero_division=0)
        
    p_val = float(np.mean(np.abs(perm_diffs) >= np.abs(obs_diff)))
    return p_val, ci_delta

if __name__ == "__main__":
    y = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1])
    p1 = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 1])
    p2 = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 0])
    p, ci = paired_bootstrap_test_f1(y, p1, p2, n_bootstraps=500)
    print(f"Permutation test: p={p:.4f}, 95% CI={ci}")
