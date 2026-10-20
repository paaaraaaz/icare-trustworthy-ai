import os
import sys
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score,
    accuracy_score, auc, precision_recall_curve, brier_score_loss
)

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if base_dir not in sys.path:
    sys.path.append(base_dir)

model_dir = os.path.join(base_dir, 'model/cnn_bigru_model')
if model_dir not in sys.path:
    sys.path.append(model_dir)

from data_loader import ICAREDataset
from data.unified_loader import get_unified_loaders, HOSPITAL_MAP
from experiments.tcn.model_tcn import ICAREModelTCN

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        return focal.mean() if self.reduction == 'mean' else focal

def compute_ece(y_true, y_prob, n_bins=15):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        if i == n_bins - 1:
            bin_idx = (y_prob >= bin_limits[i]) & (y_prob <= bin_limits[i+1])
        else:
            bin_idx = (y_prob >= bin_limits[i]) & (y_prob < bin_limits[i+1])
        if np.sum(bin_idx) > 0:
            acc = np.mean(y_true[bin_idx])
            conf = np.mean(y_prob[bin_idx])
            ece += np.sum(bin_idx) * np.abs(acc - conf)
    return float(ece / len(y_true))

def fit_temperature_scaling(val_logits, val_labels):
    def eval_nll(T):
        scaled = val_logits / max(T[0], 1e-3)
        loss = np.maximum(scaled, 0) - scaled * val_labels + np.log(1 + np.exp(-np.abs(scaled)))
        return np.mean(loss)
    res = minimize(eval_nll, [1.0], bounds=[(1e-2, 10.0)], method='L-BFGS-B')
    return float(res.x[0])

def eval_tcn(model, loader):
    model.eval()
    all_logits = []
    all_targets = []
    with torch.no_grad():
        for x_bci, x_meta, y in loader:
            x_bci = x_bci.to(DEVICE)
            x_meta = x_meta.to(DEVICE)
            logits = model(x_bci, x_meta) # shape (batch, 1)
            all_logits.extend(logits.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())
    logits_arr = np.array(all_logits)
    targets_arr = np.array(all_targets)
    probs_arr = 1.0 / (1.0 + np.exp(-logits_arr))
    return logits_arr, probs_arr, targets_arr

def run_tcn_loho_pipeline(out_dir='/Users/paaar/PycharmProjects/icare/icare_project/experiments/tuning_results_v2'):
    print(f"=== Starting 5-Fold LOHO TCN Pipeline on {DEVICE} ===")
    dataset = ICAREDataset()
    print(f"Cohort size: {len(dataset)} patients.")
    
    pooled_y = []
    pooled_probs = []
    pooled_probs_uncal = []
    pooled_logits = []
    pooled_preds = []
    fold_records = []
    
    tcn_checkpoints_dir = os.path.join(out_dir, 'tcn_rerun_checkpoints')
    os.makedirs(tcn_checkpoints_dir, exist_ok=True)
    
    for fold_idx, hosp in enumerate(HOSPITAL_MAP):
        print(f"\n--- Fold {fold_idx + 1}/5: Held-out Hospital {hosp} ---")
        train_l, val_l, test_l, _ = get_unified_loaders(dataset, fold_idx=fold_idx, batch_size=16)
        
        # Instantiate TCN with verified shape
        model = ICAREModelTCN(meta_dim=6, embed_dim=64, kernel_size=3, dropout=0.15)
        model.to(DEVICE)
        
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        
        best_val_loss = float('inf')
        best_state = None
        patience = 4
        no_improve = 0
        
        for epoch in range(20):
            model.train()
            for x_bci, x_meta, y in train_l:
                x_bci = x_bci.to(DEVICE)
                x_meta = x_meta.to(DEVICE)
                y = y.to(DEVICE)
                
                optimizer.zero_grad()
                logits = model(x_bci, x_meta)
                loss = criterion(logits, y.unsqueeze(1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for x_bci, x_meta, y in val_l:
                    x_bci, x_meta, y = x_bci.to(DEVICE), x_meta.to(DEVICE), y.to(DEVICE)
                    l = criterion(model(x_bci, x_meta), y.unsqueeze(1))
                    val_loss += l.item()
            avg_vloss = val_loss / max(len(val_l), 1)
            
            if avg_vloss < best_val_loss:
                best_val_loss = avg_vloss
                best_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break
                    
        # Load best weights
        model.load_state_dict(best_state)
        torch.save(best_state, os.path.join(tcn_checkpoints_dir, f'best_tcn_hosp_{hosp}.pt'))
        
        # 1. Validation Set Calibration and Thresholding
        val_logits, val_probs, val_y = eval_tcn(model, val_l)
        T_cal = fit_temperature_scaling(val_logits, val_y)
        fpr, tpr, thresholds = roc_curve(val_y, val_probs)
        j_scores = tpr - fpr
        th = float(thresholds[np.argmax(j_scores)])
        
        # 2. Test Set Evaluation
        test_logits, test_probs_uncal, test_y = eval_tcn(model, test_l)
        test_probs_cal = 1.0 / (1.0 + np.exp(-test_logits / T_cal))
        test_preds = (test_probs_cal >= th).astype(int)
        
        fold_auc = roc_auc_score(test_y, test_probs_cal)
        fold_f1_pos = f1_score(test_y, test_preds, average='binary', zero_division=0)
        fold_f1_mac = f1_score(test_y, test_preds, average='macro', zero_division=0)
        fold_acc = accuracy_score(test_y, test_preds)
        
        fold_records.append({
            'Fold': fold_idx, 'Hospital': hosp, 'AUROC': fold_auc,
            'F1_Positive': fold_f1_pos, 'F1_Macro': fold_f1_mac,
            'Accuracy': fold_acc, 'Threshold': th, 'T_cal': T_cal
        })
        print(f"Fold {fold_idx + 1} ({hosp}): Test AUROC={fold_auc:.3f}, F1_pos={fold_f1_pos:.3f}, F1_mac={fold_f1_mac:.3f}, Acc={fold_acc:.3f}")
        
        pooled_y.extend(test_y)
        pooled_probs.extend(test_probs_cal)
        pooled_probs_uncal.extend(test_probs_uncal)
        pooled_logits.extend(test_logits / T_cal)
        pooled_preds.extend(test_preds)
        
    y_all = np.array(pooled_y)
    p_all = np.array(pooled_probs)
    pu_all = np.array(pooled_probs_uncal)
    pred_all = np.array(pooled_preds)
    
    # Save pooled arrays in tuning_results_v2/pooled/TCN/
    tcn_pooled_dir = os.path.join(out_dir, 'pooled/TCN')
    os.makedirs(tcn_pooled_dir, exist_ok=True)
    np.save(os.path.join(tcn_pooled_dir, 'y_true.npy'), y_all)
    np.save(os.path.join(tcn_pooled_dir, 'probs.npy'), p_all)
    np.save(os.path.join(tcn_pooled_dir, 'probs_uncal.npy'), pu_all)
    np.save(os.path.join(tcn_pooled_dir, 'preds.npy'), pred_all)
    np.save(os.path.join(tcn_pooled_dir, 'logits.npy'), np.array(pooled_logits))
    
    # Also save to github repo
    tcn_github_pooled = '/Users/paaar/PycharmProjects/icare/icare_github_repo/icare_project/experiments/tuning_results_v2/pooled/TCN'
    os.makedirs(tcn_github_pooled, exist_ok=True)
    np.save(os.path.join(tcn_github_pooled, 'y_true.npy'), y_all)
    np.save(os.path.join(tcn_github_pooled, 'probs.npy'), p_all)
    np.save(os.path.join(tcn_github_pooled, 'probs_uncal.npy'), pu_all)
    np.save(os.path.join(tcn_github_pooled, 'preds.npy'), pred_all)
    np.save(os.path.join(tcn_github_pooled, 'logits.npy'), np.array(pooled_logits))
    
    # Calculate pooled overall metrics
    auroc = roc_auc_score(y_all, p_all)
    prec_c, rec_c, _ = precision_recall_curve(y_all, p_all)
    prauc = auc(rec_c, prec_c)
    f1_pos = f1_score(y_all, pred_all, average='binary', zero_division=0)
    f1_mac = f1_score(y_all, pred_all, average='macro', zero_division=0)
    acc = accuracy_score(y_all, pred_all)
    sens = recall_score(y_all, pred_all, zero_division=0)
    
    # Specificity
    tn = np.sum((y_all == 0) & (pred_all == 0))
    fp = np.sum((y_all == 0) & (pred_all == 1))
    spec = tn / max(tn + fp, 1)
    
    # TPR @ FPR <= 0.05
    fpr_all, tpr_all, _ = roc_curve(y_all, p_all)
    valid_pts = tpr_all[fpr_all <= 0.05]
    tpr_at_fpr05 = float(np.max(valid_pts)) if len(valid_pts) > 0 else 0.0
    
    ece_val = compute_ece(y_all, p_all)
    brier_val = float(brier_score_loss(y_all, p_all))
    
    print("\n=== FINAL POOLED TCN METRICS (Rerun Verified) ===")
    print(f"AUROC:            {auroc:.3f}")
    print(f"PR-AUC:           {prauc:.3f}")
    print(f"Positive F1:      {f1_pos:.3f}")
    print(f"Macro F1:         {f1_mac:.3f}")
    print(f"Accuracy:         {acc:.3f}")
    print(f"Sensitivity:      {sens:.3f}")
    print(f"Specificity:      {spec:.3f}")
    print(f"TPR @ FPR <= 0.05:{tpr_at_fpr05:.3f}")
    print(f"Calibrated ECE:   {ece_val:.3f}")
    print(f"Calibrated Brier: {brier_val:.3f}")
    
    return {
        'AUROC': auroc, 'PRAUC': prauc, 'Positive_F1': f1_pos,
        'Macro_F1': f1_mac, 'Accuracy': acc, 'Sensitivity': sens,
        'Specificity': spec, 'TPR_at_FPR05': tpr_at_fpr05,
        'ECE': ece_val, 'Brier': brier_val
    }

if __name__ == '__main__':
    run_tcn_loho_pipeline()
