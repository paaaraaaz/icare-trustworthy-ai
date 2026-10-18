import os
import sys
import copy
import random
import joblib
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, roc_curve, f1_score, precision_recall_curve, auc, brier_score_loss, accuracy_score, recall_score, precision_score
from scipy.optimize import minimize

# Add path for imports
base_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(base_dir, 'model/cnn_bigru_model'))
sys.path.append(os.path.join(base_dir, 'experiments/tcn'))
sys.path.append(base_dir)

from data_loader import ICAREDataset
from data.unified_loader import get_unified_loaders
# NOTE (v2): DemographicNet is redefined below with an nn.Dropout module so that
# dropout is controlled consistently (off during deterministic eval, on for MC).
from model_tcn import ICAREModelTCN
from features.summary_features import extract_classical_features
from stats.delong_tests import delong_roc_test
from stats.bootstrap_tests import paired_bootstrap_test_f1

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu'))
print(f"Using device: {DEVICE}")

OUT_DIR = 'experiments/tuning_results_v2'
N_MC = 50

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def enable_mc_dropout(model):
    """Keep BatchNorm in eval mode, switch only Dropout modules to train mode."""
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()

class DemographicNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.drop = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(32, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        return x

# ----------------- Dynamic Deep Models for Tuning -----------------
class TunedBCIEncoder(nn.Module):
    def __init__(self, hidden_dim=64, dropout_rate=0.2, input_channels=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=10, stride=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.gru = nn.GRU(128, hidden_dim, batch_first=True, bidirectional=True)
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        self.drop1 = nn.Dropout(dropout_rate)
        self.drop2 = nn.Dropout(dropout_rate)
        self.drop3 = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop3(x)
        x = x.permute(0, 2, 1)
        out, _ = self.gru(x)
        x = out.permute(0, 2, 1)
        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc(x))
        return x

class TunedICAREModel(nn.Module):
    def __init__(self, gru_hidden=64, dropout_rate=0.2, fusion_dropout_rate=0.2, meta_dim=6):
        super().__init__()
        self.branch_a = TunedBCIEncoder(hidden_dim=gru_hidden, dropout_rate=dropout_rate)
        self.branch_b = DemographicNet(input_dim=meta_dim, hidden_dim=32)
        self.fusion = nn.Linear(gru_hidden + 32, 32)
        self.fusion_drop = nn.Dropout(fusion_dropout_rate)
        self.output = nn.Linear(32, 1)

    def forward(self, x_bci, x_meta):
        feat_a = self.branch_a(x_bci)
        feat_b = self.branch_b(x_meta)
        combined = torch.cat([feat_a, feat_b], dim=1)
        x = F.relu(self.fusion(combined))
        x = self.fusion_drop(x)
        logits = self.output(x)
        return logits

# Focal Loss definition matching implementation
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

# ----------------- ECE and Calibration Helper -----------------
def compute_ece(y_true, y_prob, n_bins=15):
    bin_limits = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        bin_idx = (y_prob >= bin_limits[i]) & (y_prob < bin_limits[i+1])
        if np.sum(bin_idx) > 0:
            acc = np.mean(y_true[bin_idx])
            conf = np.mean(y_prob[bin_idx])
            ece += np.sum(bin_idx) * np.abs(acc - conf)
    return ece / len(y_true)

def fit_temperature_scaling(val_logits, val_labels):
    """Fits Temperature Scaling parameter T on validation set logits using NLL minimization."""
    def eval_nll(T):
        scaled_logits = val_logits / T[0]
        # binary cross entropy
        loss = np.maximum(scaled_logits, 0) - scaled_logits * val_labels + np.log(1 + np.exp(-np.abs(scaled_logits)))
        return np.mean(loss)
    res = minimize(eval_nll, [1.0], bounds=[(1e-3, 10.0)], method='L-BFGS-B')
    return res.x[0]

# ----------------- Feature Extraction for Classical Models -----------------
def extract_flat_features(loader):
    X_out, y_out = [], []
    for x_bci, x_meta, y in loader:
        for i in range(len(y)):
            seq = x_bci[i].numpy()
            meta = x_meta[i].numpy()
            feats = extract_classical_features(seq)
            X_out.append(np.concatenate([feats, meta]))
            y_out.append(y[i].item())
    return np.array(X_out), np.array(y_out)

# ----------------- Candidate Search Space Generation -----------------
def log_uniform(rng, lo, hi):
    return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))

def get_cnn_bigru_search_space(fold_idx, n_configs=8):
    rng = np.random.default_rng(seed=fold_idx * 10 + 42)
    configs = []
    for _ in range(n_configs):
        lr = log_uniform(rng, 1e-4, 3e-3)
        gru_hidden = int(rng.choice([32, 64]))
        drop = float(rng.uniform(0.10, 0.35))
        fusion_drop = float(rng.uniform(0.10, 0.30))
        configs.append({
            'lr': lr,
            'gru_hidden': gru_hidden,
            'dropout': drop,
            'fusion_dropout': fusion_drop
        })
    return configs

def get_tcn_search_space(fold_idx, n_configs=8):
    rng = np.random.default_rng(seed=fold_idx * 20 + 42)
    configs = []
    for _ in range(n_configs):
        lr = log_uniform(rng, 1e-4, 3e-3)
        drop = float(rng.uniform(0.10, 0.25))
        fusion_drop = float(rng.uniform(0.10, 0.25))
        configs.append({
            'lr': lr,
            'dropout': drop,
            'fusion_dropout': fusion_drop
        })
    return configs

def get_xgb_search_space(fold_idx, n_configs=8):
    rng = np.random.default_rng(seed=fold_idx * 30 + 42)
    configs = []
    for _ in range(n_configs):
        lr = log_uniform(rng, 0.01, 0.2)
        max_depth = int(rng.choice([3, 5, 7]))
        n_estimators = int(rng.choice([100, 300]))
        subsample = float(rng.uniform(0.6, 1.0))
        colsample_bytree = float(rng.uniform(0.6, 1.0))
        configs.append({
            'learning_rate': lr,
            'max_depth': max_depth,
            'n_estimators': n_estimators,
            'subsample': subsample,
            'colsample_bytree': colsample_bytree
        })
    return configs

def get_rf_search_space(fold_idx, n_configs=8):
    rng = np.random.default_rng(seed=fold_idx * 40 + 42)
    configs = []
    for _ in range(n_configs):
        n_estimators = int(rng.choice([100, 300]))
        max_depth = rng.choice([5, 10, None])
        if max_depth is not None:
            max_depth = int(max_depth)
        min_samples_leaf = int(rng.choice([1, 2, 4, 8]))
        max_features = rng.choice(['sqrt', 'log2', None])
        configs.append({
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'min_samples_leaf': min_samples_leaf,
            'max_features': max_features
        })
    return configs

def get_lr_grid():
    configs = []
    for C in [0.01, 0.05, 0.1, 0.5, 1.0, 5.0]:
        for penalty in ['l1', 'l2']:
            configs.append({
                'C': C,
                'penalty': penalty,
                'solver': 'liblinear'
            })
    return configs

# ----------------- Evaluation and Train Loop (v2) -----------------
def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def prob_to_logit(p):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    return np.log(p / (1 - p))

def evaluate_pytorch_model(model, loader, mc=True, num_mc_samples=N_MC):
    """Returns (logit_of_mean_prob, mean_prob, targets, passes[S, N]).
    mc=True: BatchNorm frozen, Dropout active, S stochastic passes averaged in probability space.
    mc=False: fully deterministic single pass (dropout off)."""
    if mc:
        enable_mc_dropout(model)
        S = num_mc_samples
    else:
        model.eval()
        S = 1
    batch_passes, all_targets = [], []
    with torch.no_grad():
        for x_bci, x_meta, y in loader:
            x_bci, x_meta = x_bci.to(DEVICE), x_meta.to(DEVICE)
            p = [torch.sigmoid(model(x_bci, x_meta)).cpu().numpy().flatten() for _ in range(S)]
            batch_passes.append(np.stack(p, axis=0))
            all_targets.extend(y.numpy())
    model.eval()
    passes = np.concatenate(batch_passes, axis=1)
    mean_p = passes.mean(axis=0)
    return prob_to_logit(mean_p), mean_p, np.array(all_targets), passes

def train_pytorch_model(model, train_loader, val_loader, lr, max_epochs=30, patience=5):
    model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    best_val_loss, best_state, epochs_no_improve = float('inf'), None, 0
    for epoch in range(max_epochs):
        model.train()
        for x_bci, x_meta, y in train_loader:
            x_bci, x_meta, y = x_bci.to(DEVICE), x_meta.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(x_bci, x_meta), y.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_bci, x_meta, y in val_loader:
                x_bci, x_meta, y = x_bci.to(DEVICE), x_meta.to(DEVICE), y.to(DEVICE)
                val_loss += criterion(model(x_bci, x_meta), y.unsqueeze(1)).item()
        avg_val_loss = val_loss / len(val_loader)
        if avg_val_loss < best_val_loss:
            best_val_loss, best_state, epochs_no_improve = avg_val_loss, copy.deepcopy(model.state_dict()), 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model

def calibrate_and_threshold(val_logits, y_val):
    """Temperature fitted on validation logits; Youden threshold chosen on CALIBRATED validation probs."""
    T = fit_temperature_scaling(val_logits, y_val)
    val_cal = sigmoid(val_logits / T)
    fpr, tpr, thr = roc_curve(y_val, val_cal)
    j = tpr - fpr
    return float(T), float(thr[np.argmax(j)])

def better(auc_val, macro_val, best_auc, best_macro):
    return auc_val > best_auc or (abs(auc_val - best_auc) < 1e-7 and macro_val > best_macro)

def specificity_score(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    return float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0

HOSPITAL_MAP = ['A', 'B', 'D', 'E', 'F']
MODELS = ['LogisticRegression', 'RandomForest', 'XGBoost', 'CNN-BiGRU', 'TCN']

# ----------------- Main Execution Loop -----------------
def main():
    set_seed(42)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs('tables', exist_ok=True)
    dataset = ICAREDataset('../data/BCI_Features_for_all_train_ICARE_Subjects_.mat',
                           metadata_root='../data/physionet_metadata/training')

    validation_search_records, selected_hyperparams, test_fold_records = [], [], []
    pooled = {m: {k: [] for k in ['targets', 'probs', 'probs_uncal', 'preds']} for m in MODELS}

    search_space_records = [
        {'Model': 'CNN-BiGRU', 'Hyperparameter': 'learning_rate', 'Type': 'log-uniform', 'Range': '[1e-4, 3e-3]'},
        {'Model': 'CNN-BiGRU', 'Hyperparameter': 'gru_hidden', 'Type': 'discrete', 'Range': '{32, 64}'},
        {'Model': 'CNN-BiGRU', 'Hyperparameter': 'dropout', 'Type': 'uniform', 'Range': '[0.10, 0.35]'},
        {'Model': 'CNN-BiGRU', 'Hyperparameter': 'fusion_dropout', 'Type': 'uniform', 'Range': '[0.10, 0.30]'},
        {'Model': 'CNN-BiGRU', 'Hyperparameter': 'n_configs', 'Type': 'fixed', 'Range': '8'},
        {'Model': 'TCN', 'Hyperparameter': 'learning_rate', 'Type': 'log-uniform', 'Range': '[1e-4, 3e-3]'},
        {'Model': 'TCN', 'Hyperparameter': 'dropout', 'Type': 'uniform', 'Range': '[0.10, 0.25]'},
        {'Model': 'TCN', 'Hyperparameter': 'fusion_dropout', 'Type': 'uniform', 'Range': '[0.10, 0.25]'},
        {'Model': 'TCN', 'Hyperparameter': 'n_configs', 'Type': 'fixed', 'Range': '8'},
        {'Model': 'XGBoost', 'Hyperparameter': 'learning_rate', 'Type': 'log-uniform', 'Range': '[0.01, 0.2]'},
        {'Model': 'XGBoost', 'Hyperparameter': 'max_depth', 'Type': 'discrete', 'Range': '{3, 5, 7}'},
        {'Model': 'XGBoost', 'Hyperparameter': 'n_estimators', 'Type': 'discrete', 'Range': '{100, 300}'},
        {'Model': 'XGBoost', 'Hyperparameter': 'subsample', 'Type': 'uniform', 'Range': '[0.6, 1.0]'},
        {'Model': 'XGBoost', 'Hyperparameter': 'colsample_bytree', 'Type': 'uniform', 'Range': '[0.6, 1.0]'},
        {'Model': 'RandomForest', 'Hyperparameter': 'n_estimators', 'Type': 'discrete', 'Range': '{100, 300}'},
        {'Model': 'RandomForest', 'Hyperparameter': 'max_depth', 'Type': 'discrete', 'Range': '{5, 10, None}'},
        {'Model': 'RandomForest', 'Hyperparameter': 'min_samples_leaf', 'Type': 'discrete', 'Range': '{1, 2, 4, 8}'},
        {'Model': 'RandomForest', 'Hyperparameter': 'max_features', 'Type': 'discrete', 'Range': '{sqrt, log2, None}'},
        {'Model': 'LogisticRegression', 'Hyperparameter': 'C', 'Type': 'discrete', 'Range': '{0.01, 0.05, 0.1, 0.5, 1.0, 5.0}'},
        {'Model': 'LogisticRegression', 'Hyperparameter': 'penalty', 'Type': 'discrete', 'Range': '{l1, l2}'},
    ]
    pd.DataFrame(search_space_records).to_csv(os.path.join(OUT_DIR, 'search_space_by_model.csv'), index=False)

    for fold_idx in range(5):
        print(f"\n================ FOLD {fold_idx} ================", flush=True)
        set_seed(1000 * fold_idx)
        train_l, val_l, test_l, hosp = get_unified_loaders(dataset, fold_idx, batch_size=32)
        tr_idx, va_idx, te_idx = set(train_l.dataset.indices), set(val_l.dataset.indices), set(test_l.dataset.indices)
        assert not (tr_idx & va_idx) and not (tr_idx & te_idx) and not (va_idx & te_idx), "Partition overlap!"
        assert all(dataset.hospitals[i] == hosp for i in te_idx)
        assert all(dataset.hospitals[i] != hosp for i in tr_idx | va_idx)
        print(f"Test hospital: {hosp} | train {len(tr_idx)} val {len(va_idx)} test {len(te_idx)}", flush=True)

        # Classical features from the PLAIN (non-resampled) training subset
        plain_train_l = DataLoader(train_l.dataset, batch_size=32, shuffle=False)
        X_train, y_train = extract_flat_features(plain_train_l)
        X_val, y_val = extract_flat_features(val_l)
        X_test, y_test = extract_flat_features(test_l)
        X_train, X_val, X_test = map(np.nan_to_num, (X_train, X_val, X_test))
        scaler = StandardScaler().fit(X_train)
        X_train_s, X_val_s, X_test_s = scaler.transform(X_train), scaler.transform(X_val), scaler.transform(X_test)
        joblib.dump(scaler, os.path.join(OUT_DIR, f'fold_{fold_idx}_scaler.joblib'))

        fold_out = {}

        # ---------- Classical models ----------
        classical = {
            'LogisticRegression': (get_lr_grid(), lambda c: LogisticRegression(C=c['C'], penalty=c['penalty'], solver=c['solver'], class_weight='balanced', max_iter=2000)),
            'RandomForest': (get_rf_search_space(fold_idx), lambda c: RandomForestClassifier(n_estimators=c['n_estimators'], max_depth=c['max_depth'], min_samples_leaf=c['min_samples_leaf'], max_features=c['max_features'], class_weight='balanced', random_state=42, n_jobs=-1)),
            'XGBoost': (get_xgb_search_space(fold_idx), lambda c: XGBClassifier(learning_rate=c['learning_rate'], max_depth=c['max_depth'], n_estimators=c['n_estimators'], subsample=c['subsample'], colsample_bytree=c['colsample_bytree'], random_state=42, eval_metric='logloss')),
        }
        for name, (configs, make) in classical.items():
            print(f"{name} tuning...", flush=True)
            best_auc, best_macro, best_cfg = -1.0, -1.0, None
            for idx, cfg in enumerate(configs):
                mdl = make(cfg).fit(X_train_s, y_train)
                vp = mdl.predict_proba(X_val_s)[:, 1]
                a, mf = roc_auc_score(y_val, vp), f1_score(y_val, (vp > 0.5).astype(int), average='macro')
                validation_search_records.append({'Fold': fold_idx, 'Model': name, 'Config_Idx': idx, 'Config_Details': str(cfg), 'Val_AUROC': a, 'Val_Macro_F1': mf})
                if better(a, mf, best_auc, best_macro):
                    best_auc, best_macro, best_cfg = a, mf, cfg
            final = make(best_cfg).fit(X_train_s, y_train)
            mdir = os.path.join(OUT_DIR, f'fold_{fold_idx}', name); os.makedirs(mdir, exist_ok=True)
            joblib.dump(final, os.path.join(mdir, 'model.joblib'))
            T, th = calibrate_and_threshold(prob_to_logit(final.predict_proba(X_val_s)[:, 1]), y_val)
            test_uncal = final.predict_proba(X_test_s)[:, 1]
            fold_out[name] = dict(cfg=best_cfg, T=T, th=th, uncal=test_uncal, cal=sigmoid(prob_to_logit(test_uncal) / T), y=y_test, passes=None)

        # ---------- Deep models ----------
        def make_cnn(c):
            return TunedICAREModel(gru_hidden=c['gru_hidden'], dropout_rate=c['dropout'], fusion_dropout_rate=c['fusion_dropout'])
        def make_tcn(c):
            m = ICAREModelTCN(meta_dim=6, embed_dim=64, kernel_size=3, dropout=c['dropout'])
            m.fusion_dropout.p = c['fusion_dropout']
            return m
        deep = {'CNN-BiGRU': (get_cnn_bigru_search_space(fold_idx), make_cnn),
                'TCN': (get_tcn_search_space(fold_idx), make_tcn)}
        for name, (configs, make) in deep.items():
            print(f"{name} tuning...", flush=True)
            best_auc, best_macro, best_cfg, best_state = -1.0, -1.0, None, None
            for idx, cfg in enumerate(configs):
                set_seed(1000 * fold_idx + 100 * (name == 'TCN') + idx)
                mdl = train_pytorch_model(make(cfg), train_l, val_l, cfg['lr'])
                _, vp, _, _ = evaluate_pytorch_model(mdl, val_l, mc=False)
                a, mf = roc_auc_score(y_val, vp), f1_score(y_val, (vp > 0.5).astype(int), average='macro')
                print(f"  cfg {idx}: val AUROC {a:.4f}", flush=True)
                validation_search_records.append({'Fold': fold_idx, 'Model': name, 'Config_Idx': idx, 'Config_Details': str(cfg), 'Val_AUROC': a, 'Val_Macro_F1': mf})
                if better(a, mf, best_auc, best_macro):
                    best_auc, best_macro, best_cfg, best_state = a, mf, cfg, copy.deepcopy(mdl.state_dict())
            final = make(best_cfg); final.load_state_dict(best_state); final.to(DEVICE)
            mdir = os.path.join(OUT_DIR, f'fold_{fold_idx}', name); os.makedirs(mdir, exist_ok=True)
            torch.save(best_state, os.path.join(mdir, 'best.pth'))
            set_seed(7 + fold_idx)
            val_logits, _, _, _ = evaluate_pytorch_model(final, val_l, mc=True)
            T, th = calibrate_and_threshold(val_logits, y_val)
            test_logits, test_mean, yt, passes = evaluate_pytorch_model(final, test_l, mc=True)
            assert np.array_equal(yt, y_test)
            np.save(os.path.join(mdir, 'mc_passes_uncal.npy'), passes)
            fold_out[name] = dict(cfg=best_cfg, T=T, th=th, uncal=test_mean, cal=sigmoid(test_logits / T), y=y_test, passes=passes)

        # ---------- Save fold outputs ----------
        for m in MODELS:
            o = fold_out[m]
            preds = (o['cal'] > o['th']).astype(int)
            mdir = os.path.join(OUT_DIR, f'fold_{fold_idx}', m); os.makedirs(mdir, exist_ok=True)
            np.save(os.path.join(mdir, 'y_true.npy'), o['y'])
            np.save(os.path.join(mdir, 'probs.npy'), o['cal'])
            np.save(os.path.join(mdir, 'probs_uncal.npy'), o['uncal'])
            np.save(os.path.join(mdir, 'logits.npy'), prob_to_logit(o['cal']))
            np.save(os.path.join(mdir, 'preds.npy'), preds)
            with open(os.path.join(mdir, 'chosen_threshold.txt'), 'w') as f:
                f.write(f"{o['th']}\n")
            selected_hyperparams.append({'Fold': fold_idx, 'Hospital': hosp, 'Model': m, 'Config': str(o['cfg']), 'Threshold': o['th'], 'T_cal': o['T']})
            test_fold_records.append({
                'Fold': fold_idx, 'Hospital': hosp, 'Model': m, 'N': len(o['y']),
                'AUROC': roc_auc_score(o['y'], o['cal']), 'F1': f1_score(o['y'], preds, zero_division=0),
                'Accuracy': accuracy_score(o['y'], preds), 'Sensitivity': recall_score(o['y'], preds, zero_division=0),
                'Specificity': specificity_score(o['y'], preds), 'Precision': precision_score(o['y'], preds, zero_division=0),
                'ECE_pre': compute_ece(o['y'], o['uncal']), 'ECE_post': compute_ece(o['y'], o['cal']),
                'Brier_pre': brier_score_loss(o['y'], o['uncal']), 'Brier_post': brier_score_loss(o['y'], o['cal']),
                'Threshold': o['th'], 'T_cal': o['T']})
            pooled[m]['targets'].extend(o['y']); pooled[m]['probs'].extend(o['cal'])
            pooled[m]['probs_uncal'].extend(o['uncal']); pooled[m]['preds'].extend(preds)

        # checkpoint progress after every fold
        pd.DataFrame(validation_search_records).to_csv(os.path.join(OUT_DIR, 'all_validation_search_results.csv'), index=False)
        pd.DataFrame(selected_hyperparams).to_csv(os.path.join(OUT_DIR, 'selected_hyperparameters_by_outer_fold.csv'), index=False)
        pd.DataFrame(test_fold_records).to_csv(os.path.join(OUT_DIR, 'tuned_test_results_by_fold.csv'), index=False)

    # ---------- Pooled ----------
    rows = []
    for m in MODELS:
        y = np.array(pooled[m]['targets']); p = np.array(pooled[m]['probs'])
        pu = np.array(pooled[m]['probs_uncal']); pr = np.array(pooled[m]['preds'])
        prec_c, rec_c, _ = precision_recall_curve(y, p)
        d = os.path.join(OUT_DIR, 'pooled', m); os.makedirs(d, exist_ok=True)
        for k, v in [('y_true', y), ('probs', p), ('probs_uncal', pu), ('preds', pr), ('logits', prob_to_logit(p))]:
            np.save(os.path.join(d, f'{k}.npy'), v)
        rows.append({'Model': m, 'Pooled_AUROC': roc_auc_score(y, p), 'Pooled_PRAUC': auc(rec_c, prec_c),
                     'Pooled_F1': f1_score(y, pr), 'Pooled_Accuracy': accuracy_score(y, pr),
                     'Pooled_Sensitivity': recall_score(y, pr), 'Pooled_Specificity': specificity_score(y, pr),
                     'Pooled_ECE_pre': compute_ece(y, pu), 'Pooled_ECE_post': compute_ece(y, p),
                     'Pooled_Brier_pre': brier_score_loss(y, pu), 'Pooled_Brier_post': brier_score_loss(y, p)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, 'pooled_tuned_performance_table.csv'), index=False)
    df.to_csv('tables/pooled_tuned_performance_table_v2.csv', index=False)
    print(df.to_string(), flush=True)

    cy = np.array(pooled['CNN-BiGRU']['targets']); cp = np.array(pooled['CNN-BiGRU']['probs']); cpr = np.array(pooled['CNN-BiGRU']['preds'])
    comp = []
    for m in ['LogisticRegression', 'RandomForest', 'XGBoost', 'TCN']:
        p_delong, _ = delong_roc_test(cy, cp, np.array(pooled[m]['probs']))
        p_boot, _ = paired_bootstrap_test_f1(cy, cpr, np.array(pooled[m]['preds']), n_bootstraps=1000)
        comp.append({'Comparison': f'CNN-BiGRU vs {m}', 'DeLong_p_value': p_delong, 'Bootstrap_F1_p_value': p_boot})
    pd.DataFrame(comp).to_csv(os.path.join(OUT_DIR, 'tuned_pairwise_comparisons.csv'), index=False)
    print(pd.DataFrame(comp).to_string(), flush=True)

if __name__ == '__main__':
    main()
