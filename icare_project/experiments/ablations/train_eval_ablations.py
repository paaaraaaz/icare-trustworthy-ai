import os
import sys
import copy
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score, accuracy_score, auc, precision_recall_curve
)

# Robust path handling
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if base_dir not in sys.path:
    sys.path.append(base_dir)

model_dir = os.path.join(base_dir, 'model/cnn_bigru_model')
if model_dir not in sys.path:
    sys.path.append(model_dir)

from data_loader import ICAREDataset
from data.unified_loader import get_unified_loaders, HOSPITAL_MAP
from experiments.ablations.model_ablations import ICAREModelAblation

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        bce_loss = nn.functional.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss

def calculate_metrics(y_true, y_score, threshold):
    y_pred = (y_score >= threshold).astype(int)
    try:
        auc_val = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc_val = 0.5
    f1_pos = float(f1_score(y_true, y_pred, average='binary', zero_division=0))
    f1_mac = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    acc = float(accuracy_score(y_true, y_pred))
    prec = float(precision_score(y_true, y_pred, average='binary', zero_division=0))
    rec = float(recall_score(y_true, y_pred, average='binary', zero_division=0))
    return auc_val, f1_pos, f1_mac, acc, prec, rec

def evaluate_fold(model, loader, loss_type='focal', device=DEVICE):
    model.eval()
    criterion = FocalLoss(alpha=0.25, gamma=2.0) if loss_type == 'focal' else nn.BCEWithLogitsLoss()
    
    total_loss = 0.0
    all_targets = []
    all_preds = []
    
    with torch.no_grad():
        for x_bci, x_meta, y in loader:
            x_bci = x_bci.to(device)
            x_meta = x_meta.to(device)
            y = y.to(device)
            
            logits = model(x_bci, x_meta)
            loss = criterion(logits, y.unsqueeze(1))
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            all_preds.extend(probs.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy().flatten())
            
    metrics = {
        'loss': total_loss / max(len(loader), 1),
        'targets': np.array(all_targets),
        'preds': np.array(all_preds)
    }
    return metrics

def train_and_validate_ablation(model, train_loader, val_loader, test_loader,
                                variant_name='Full', hospital_id='Unknown',
                                loss_type='focal', threshold_type='youden',
                                epochs=15, lr=1e-3, device=DEVICE,
                                checkpoint_dir='checkpoints_ablations',
                                eval_only=False):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = FocalLoss(alpha=0.25, gamma=2.0) if loss_type == 'focal' else nn.BCEWithLogitsLoss()
    
    best_val_loss = float('inf')
    best_state = None
    best_model_path = os.path.join(checkpoint_dir, f'best_{variant_name}_hosp_{hospital_id}.pt')
    
    # Checkpoint loading vs fresh retraining:
    # If eval_only=True, load saved checkpoint to reproduce or audit without retraining.
    # Otherwise, always retrain fresh to guarantee reproducibility and prevent stale weight leakage.
    if eval_only:
        if os.path.exists(best_model_path):
            best_state = torch.load(best_model_path, map_location=device)
            model.load_state_dict(best_state)
        else:
            raise FileNotFoundError(f"Checkpoint not found for --eval_only: {best_model_path}")
    else:
        for epoch in range(epochs):
            model.train()
            for x_bci, x_meta, y in train_loader:
                x_bci, x_meta, y = x_bci.to(device), x_meta.to(device), y.to(device)
                optimizer.zero_grad()
                logits = model(x_bci, x_meta)
                loss = criterion(logits, y.unsqueeze(1))
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
            val_m = evaluate_fold(model, val_loader, loss_type=loss_type, device=device)
            if val_m['loss'] < best_val_loss:
                best_val_loss = val_m['loss']
                best_state = copy.deepcopy(model.state_dict())
                torch.save(best_state, best_model_path)
                
        if best_state is not None:
            model.load_state_dict(best_state)
        
    # 1. Validation set threshold tuning (completely independent of test set)
    val_m = evaluate_fold(model, val_loader, loss_type=loss_type, device=device)
    threshold = 0.5
    if threshold_type == 'youden' and len(np.unique(val_m['targets'])) > 1:
        fpr, tpr, thresholds = roc_curve(val_m['targets'], val_m['preds'])
        j_scores = tpr - fpr
        threshold = float(thresholds[np.argmax(j_scores)])
        
    # 2. Test set evaluation using frozen validation threshold
    test_m = evaluate_fold(model, test_loader, loss_type=loss_type, device=device)
    auc_val, f1_pos, f1_mac, acc, prec, rec = calculate_metrics(test_m['targets'], test_m['preds'], threshold)
    test_bin = (test_m['preds'] >= threshold).astype(int)
    
    return {
        'Variant': variant_name,
        'Hospital': hospital_id,
        'AUROC': auc_val,
        'F1_Positive': f1_pos,
        'F1_Macro': f1_mac,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'Threshold': threshold
    }, test_m['targets'], test_m['preds'], test_bin

def run_all_ablations(output_csv='ablation_table2.csv', eval_only=False, checkpoint_dir='checkpoints_ablations'):
    print(f"Loading I-CARE dataset on device: {DEVICE}...")
    # Supports optional environment variable ICARE_DATA_DIR
    dataset = ICAREDataset()
    print(f"Total cohort: {len(dataset)} patients across {len(HOSPITAL_MAP)} hospitals.")
    
    variants = {
        'Full Model': dict(use_gru=True, use_cnn=True, use_meta=True),
        'No Clinical Meta': dict(use_gru=True, use_cnn=True, use_meta=False),
        'CNN-only': dict(use_gru=False, use_cnn=True, use_meta=True),
        'GRU-only': dict(use_gru=True, use_cnn=False, use_meta=True)
    }
    
    records = []
    # Store predictions and leak-free validation-thresholded decisions
    pooled_by_variant = {v: {'targets': [], 'preds': [], 'pred_bin': []} for v in variants}
    
    for fold_idx, hosp in enumerate(HOSPITAL_MAP):
        print(f"\n================ Outer Fold {fold_idx + 1}/5: Held-out Hospital {hosp} ================")
        train_l, val_l, test_l, _ = get_unified_loaders(dataset, fold_idx=fold_idx, batch_size=16)
        
        for vname, vcfg in variants.items():
            print(f"  Evaluating Variant: {vname:18s} ...", end=" ", flush=True)
            model = ICAREModelAblation(**vcfg)
            res, y_true, y_preds, y_bin = train_and_validate_ablation(
                model, train_l, val_l, test_l,
                variant_name=vname.replace(' ', '_'),
                hospital_id=hosp,
                epochs=12,
                device=DEVICE,
                checkpoint_dir=checkpoint_dir,
                eval_only=eval_only
            )
            records.append(res)
            pooled_by_variant[vname]['targets'].extend(y_true)
            pooled_by_variant[vname]['preds'].extend(y_preds)
            pooled_by_variant[vname]['pred_bin'].extend(y_bin)
            print(f"Done. Test AUROC={res['AUROC']:.3f}, F1_pos={res['F1_Positive']:.3f}, F1_macro={res['F1_Macro']:.3f} (th={res['Threshold']:.3f})")
            
    # Compute Pooled Metrics strictly without test-label threshold leakage
    summary_rows = []
    for vname in variants:
        y_all = np.array(pooled_by_variant[vname]['targets'])
        p_all = np.array(pooled_by_variant[vname]['preds'])
        pred_bin = np.array(pooled_by_variant[vname]['pred_bin'])
        
        # 1. Threshold-independent metrics (evaluated directly on pooled probabilities)
        auroc = roc_auc_score(y_all, p_all)
        prec_c, rec_c, _ = precision_recall_curve(y_all, p_all)
        prauc = auc(rec_c, prec_c)
        
        # 2. Threshold-dependent metrics (evaluated using each fold's frozen validation threshold)
        f1_pos = f1_score(y_all, pred_bin, average='binary', zero_division=0)
        f1_mac = f1_score(y_all, pred_bin, average='macro', zero_division=0)
        acc = accuracy_score(y_all, pred_bin)
        
        summary_rows.append({
            'Ablation_Variant': vname,
            'Pooled_AUROC': round(auroc, 3),
            'Pooled_PRAUC': round(prauc, 3),
            'F1_Positive': round(f1_pos, 3),
            'F1_Macro': round(f1_mac, 3),
            'Accuracy': round(acc, 3),
            'Threshold_Protocol': 'Fold-wise Validation Youden (Leak-Free)'
        })
        
    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(output_csv, index=False)
    print("\n================ FINAL STRICTLY LEAK-FREE ABLATION TABLE 2 ================")
    print(df_summary.to_string(index=False))
    return df_summary

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Strictly leak-free ablation study runner.")
    parser.add_argument('--output_csv', type=str, default='ablation_table2.csv', help="Output CSV path for results")
    parser.add_argument('--eval_only', action='store_true', help="Evaluate existing checkpoints instead of retraining fresh")
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints_ablations', help="Directory to store or load checkpoints")
    args = parser.parse_args()
    run_all_ablations(output_csv=args.output_csv, eval_only=args.eval_only, checkpoint_dir=args.checkpoint_dir)
