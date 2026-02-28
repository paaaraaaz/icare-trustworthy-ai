import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_score, recall_score

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
    y_pred = (y_score > threshold).astype(int)
    try:
        auc = roc_auc_score(y_true, y_score)
    except:
        auc = 0.0
    f1 = f1_score(y_true, y_pred, zero_division=0)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    return auc, f1, prec, rec

def evaluate_fold(model, val_loader, loss_type='focal', device='cpu'):
    model.eval()
    
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    val_loss = 0
    all_targets = []
    all_preds_mean = []
    
    with torch.no_grad():
        for x_bci, x_meta, y in val_loader:
            x_bci, x_meta, y = x_bci.to(device), x_meta.to(device), y.to(device)
            logits = model(x_bci, x_meta)
            loss = criterion(logits, y.unsqueeze(1))
            val_loss += loss.item()
            
            probs = torch.sigmoid(logits)
            all_preds_mean.extend(probs.cpu().numpy().flatten())
            all_targets.extend(y.cpu().numpy())
            
    metrics = {}
    if len(val_loader) > 0:
        metrics['loss'] = val_loss / len(val_loader)
        
    metrics['targets'] = np.array(all_targets)
    metrics['preds'] = np.array(all_preds_mean)
    return metrics

def train_and_validate_ablation(model, train_loader, val_loader, test_loader, 
                               loss_type='focal', threshold_type='youden',
                               epochs=10, lr=1e-3, device='cpu', 
                               hospital_id='Unknown', checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    if loss_type == 'focal':
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
    else:
        criterion = nn.BCEWithLogitsLoss()
        
    best_val_loss = float('inf')
    best_model_path = os.path.join(checkpoint_dir, f"best_model_hospital_{hospital_id}.pth")
    
    for epoch in range(epochs):
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
        
        # Execute deterministic validation pass to calculate independent fold metrics.
        val_metrics = evaluate_fold(model, val_loader, loss_type=loss_type, device=device)
        avg_val_loss = val_metrics['loss']
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            
    # Load best model for evaluation
    model.load_state_dict(torch.load(best_model_path))
    
    # Step 1: Compute the optimal decision threshold dynamically based on validation set Youden Index.
    val_metrics = evaluate_fold(model, val_loader, loss_type=loss_type, device=device)
    val_targets = val_metrics['targets']
    val_preds = val_metrics['preds']
    
    threshold = 0.5
    if threshold_type == 'youden':
        fpr, tpr, thresholds = roc_curve(val_targets, val_preds)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        threshold = thresholds[best_idx]
        
    # Step 2: Perform final independent evaluation on the strictly held-out test dataset using the optimal threshold.
    test_metrics = evaluate_fold(model, test_loader, loss_type=loss_type, device=device)
    test_targets = test_metrics['targets']
    test_preds = test_metrics['preds']
    
    auc, f1, prec, rec = calculate_metrics(test_targets, test_preds, threshold)
    
    return {
        'Hospital': hospital_id,
        'AUC': auc,
        'F1': f1,
        'Precision': prec,
        'Recall': rec,
        'OptimalThreshold': float(threshold),
        'LossType': loss_type,
        'ThresholdType': threshold_type
    }, test_targets, test_preds
