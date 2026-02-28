
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
import os

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

def train_and_validate(model, train_loader, val_loader, epochs=10, lr=1e-3, device='cpu', hospital_id='Unknown', checkpoint_dir='model_checkpoints', trial_name=''):
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss(alpha=0.25, gamma=2.0) # Utilize Focal Loss to dynamically scale cross-entropy based on prediction confidence.
    
    best_val_loss = float('inf')
    
    prefix = f"{trial_name}_" if trial_name else ""
    best_model_path = os.path.join(checkpoint_dir, f"{prefix}best_model_hospital_{hospital_id}.pth")
    
    history = []
    
    for epoch in range(epochs):
        # --- Training ---
        model.train()
        train_loss = 0
        for x_bci, x_meta, y in train_loader:
            x_bci, x_meta, y = x_bci.to(device), x_meta.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits = model(x_bci, x_meta)
            loss = criterion(logits, y.unsqueeze(1))
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_loader)
        
        # --- Validation ---
        val_metrics = evaluate_fold(model, val_loader, device=device, num_mc_samples=1) # Fast val (no MC)
        avg_val_loss = val_metrics['loss']
        val_auc = val_metrics.get('auc', 0.0)
        
        print(f"Epoch {epoch+1}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val AUC: {val_auc:.4f}")
        
        history.append({
            'Hospital': hospital_id,
            'Epoch': epoch + 1,
            'Train Loss': avg_train_loss,
            'Val Loss': avg_val_loss,
            'Val AUC': val_auc
        })
        
        # --- Checkpointing ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            # print(f"  Saved best model to {best_model_path}")
            
    # Load Best Model for Final Evaluation
    model.load_state_dict(torch.load(best_model_path))
    return model, history

def evaluate_fold(model, val_loader, device='cpu', num_mc_samples=50):
    model.eval()
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    val_loss = 0
    all_targets = []
    all_preds_mean = []
    all_preds_entropy = []
    all_metas = []
    
    with torch.no_grad():
        for x_bci, x_meta, y in val_loader:
            x_bci, x_meta, y = x_bci.to(device), x_meta.to(device), y.to(device)
            
            # Calculate loss using a single deterministic forward pass for efficiency during standard validation.
            # Fast validation operations utilize num_mc_samples = 1.
            
            batch_preds = []
            batch_logits = []
            
            for _ in range(num_mc_samples):
                logits = model(x_bci, x_meta)
                probs = torch.sigmoid(logits)
                batch_preds.append(probs.cpu().numpy())
                batch_logits.append(logits)
                
            # Compute Loss on Mean Logits (approx) or Mean Probs
            # BCEWithLogitsLoss needs logits.
            # Let's just take the first pass capability for loss if num_mc=1
            if num_mc_samples == 1:
                loss = criterion(batch_logits[0], y.unsqueeze(1))
                val_loss += loss.item()
            else:
                 # For Monte Carlo (MC) sampling, the calculation focuses exclusively on aggregating predictive probabilities and entropy.
                 pass

            batch_preds = np.array(batch_preds)
            mean_pred = np.mean(batch_preds, axis=0)
            
            # Entropy
            epsilon = 1e-10
            p = mean_pred
            entropy = - (p * np.log(p + epsilon) + (1 - p) * np.log(1 - p + epsilon))
            
            all_preds_mean.extend(mean_pred.flatten())
            all_preds_entropy.extend(entropy.flatten())
            all_targets.extend(y.cpu().numpy())
            all_metas.extend(x_meta.cpu().numpy())
            
    metrics = {}
    if len(val_loader) > 0 and num_mc_samples == 1:
        metrics['loss'] = val_loss / len(val_loader)
        
    try:
        metrics['auc'] = roc_auc_score(all_targets, all_preds_mean)
    except:
        metrics['auc'] = 0.0

    return {
        'loss': metrics.get('loss', 0.0), # might be 0 for final eval
        'auc': metrics['auc'],
        'targets': np.array(all_targets),
        'preds': np.array(all_preds_mean),
        'entropy': np.array(all_preds_entropy),
        'metas': np.array(all_metas)
    }

def aggregate_results(fold_results):
    # Combine all folds for global metrics
    global_targets = np.concatenate([res['targets'] for res in fold_results])
    global_preds = np.concatenate([res['preds'] for res in fold_results])
    global_entropy = np.concatenate([res['entropy'] for res in fold_results])
    global_metas = np.concatenate([res['metas'] for res in fold_results])
    
    # Global AUC
    try:
        auc = roc_auc_score(global_targets, global_preds)
    except:
        auc = 0.0
    
    # Challenge Score
    fpr, tpr, thresholds = roc_curve(global_targets, global_preds)
    eligible_indices = np.where(fpr <= 0.05)[0]
    if len(eligible_indices) > 0:
        challenge_score = np.max(tpr[eligible_indices])
    else:
        challenge_score = 0.0
        
    # Find Optimal Youden Index Threshold
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    best_threshold = thresholds[best_idx]
        
    # Subgroup Analysis
    # Meta: [Age, Sex, ROSC, OHCA, Shock, TTM]
    ages = global_metas[:, 0] # Normalized Age
    sexes = global_metas[:, 1] # 0=F, 1=M
    
    subgroups = {
        'Age < 60': ages < 0.60,
        'Age >= 60': ages >= 0.60,
        'Male': sexes == 1,
        'Female': sexes == 0
    }
    
    subgroup_res = {}
    for name, mask in subgroups.items():
        if np.sum(mask) == 0:
            subgroup_res[name] = {'AUC': float('nan'), 'FPR': float('nan'), 'Count': 0}
            continue
            
        y_true = global_targets[mask]
        y_score = global_preds[mask]
        y_pred = (y_score > best_threshold).astype(int)
        
        # Subgroup AUC
        try:
            sg_auc = roc_auc_score(y_true, y_score)
        except:
            sg_auc = float('nan') 
            
        # FPR
        fp = np.sum((y_pred == 1) & (y_true == 0))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        subgroup_res[name] = {
            'AUC': sg_auc, 
            'FPR': fpr_val, 
            'Count': int(np.sum(mask))
        }
        
    return {
        'AUC': auc,
        'ChallengeScore': challenge_score,
        'OptimalThreshold': float(best_threshold),
        'UncertaintyEntropy': float(np.mean(global_entropy)),
        'Subgroups': subgroup_res
    }
