
import torch
import numpy as np
import pandas as pd
import json
import os

from icare_project.model.cnn_bigru_model.data_loader import ICAREDataset, get_loocv_loaders, get_loocv_loaders_tri_split
from icare_project.model.cnn_bigru_model.model import ICAREModel
from icare_project.model.cnn_bigru_model.train_eval import train_and_validate, evaluate_fold, aggregate_results

def main():
    print("=== I-CARE BCI Neuroprognostication Framework (Trial 6: Focal Loss + Youden Threshold) ===")
    
    # 1. Data Setup
    file_path = 'data/BCI_Features_for_all_train_ICARE_Subjects_.mat'
    meta_path = 'data/physionet_metadata/training'
    BATCH_SIZE = 32
    EPOCHS = 10 
    TRIAL_NAME = "cnn_bigru_model"
    
    print(f"Step 1: Loading Data & Real Metadata ({TRIAL_NAME})...")
    dataset = ICAREDataset(file_path, metadata_root=meta_path, window_hours=72)
    print(f"Total Subjects Loaded: {len(dataset)}")
    
    hospitals = sorted(list(set(dataset.hospitals)))
    print(f"Hospitals found for LOOCV: {hospitals}")
    
    fold_results = []
    all_training_logs = []
    all_test_results = []
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Ensure directories exist
    os.makedirs('icare_project/model/cnn_bigru_model/logs', exist_ok=True)
    os.makedirs('icare_project/model/cnn_bigru_model/model_checkpoints', exist_ok=True)
    
    # 2. LOOCV Loop
    print("\nStep 2: Starting Leave-One-Center-Out Cross-Validation...")
    
    for val_hospital in hospitals:
        # The designated validation hospital serves as the strictly held-out test cohort for the current cross-validation fold.
        print(f"\n--- Fold: Test on Hospital {val_hospital} ---")
        
        # Get Loaders (Train / Val / Test)
        train_loader, val_loader, test_loader = get_loocv_loaders_tri_split(
            dataset, 
            leave_out_hospital=val_hospital, 
            batch_size=BATCH_SIZE,
            val_split=0.2
        )
        
        if train_loader is None:
            continue
            
        print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)} | Test batches: {len(test_loader)}")
        
        # Initialize Model (Reset for each fold)
        model = ICAREModel(meta_dim=6)
        
        # Train with Validation & Checkpointing (Internal Val)
        best_model, history = train_and_validate(
            model, 
            train_loader, 
            val_loader, 
            epochs=EPOCHS, 
            lr=0.001, 
            device=device,
            hospital_id=val_hospital, # Using test hospital ID for checkpoint naming
            checkpoint_dir='icare_project/model/cnn_bigru_model/model_checkpoints',
            trial_name=TRIAL_NAME
        )
        
        # Log Training History
        all_training_logs.extend(history)
        
        # Evaluate the highest-performing empirical model checkpoint on the strictly independent test cohort.
        res = evaluate_fold(best_model, test_loader, device=device, num_mc_samples=50)
        fold_results.append(res)
        
        # Log Test Results (Subject-level)
        # Identify Test Subjects
        test_indices = [i for i, h in enumerate(dataset.hospitals) if h == val_hospital]
        test_subjs = [dataset.subjects[i] for i in test_indices]
        
        # Verify length matches
        if len(test_subjs) == len(res['preds']):
            for i, subj in enumerate(test_subjs):
                all_test_results.append({
                    'SubjectID': subj,
                    'Hospital': val_hospital,
                    'Target': res['targets'][i],
                    'Prediction_Mean': res['preds'][i],
                    'Uncertainty_Entropy': res['entropy'][i],
                    'Age_Norm': res['metas'][i][0],
                    'Sex': res['metas'][i][1], # 0=F, 1=M
                    'ROSC_Norm': res['metas'][i][2],
                    'OHCA': res['metas'][i][3],
                    'Shockable': res['metas'][i][4],
                    'TTM': res['metas'][i][5]
                })
        
    # 3. Aggregation & Saving
    print("\nStep 3: Aggregating Results & Generating Logs...")
    final_metrics = aggregate_results(fold_results)
    
    # Save Training Logs
    pd.DataFrame(all_training_logs).to_csv(f'icare_project/model/cnn_bigru_model/logs/{TRIAL_NAME}_training_log.csv', index=False)
    print(f"Saved icare_project/model/cnn_bigru_model/logs/{TRIAL_NAME}_training_log.csv")
    
    # Save Test Results
    pd.DataFrame(all_test_results).to_csv(f'icare_project/model/cnn_bigru_model/logs/{TRIAL_NAME}_test_results.csv', index=False)
    print(f"Saved icare_project/model/cnn_bigru_model/logs/{TRIAL_NAME}_test_results.csv")
    
    print("\n=== Final Report (Aggregated LOOCV) ===")
    print(f"AUC: {final_metrics['AUC']:.4f}")
    print(f"PhysioNet Challenge Score (TPR @ FPR<=0.05): {final_metrics['ChallengeScore']:.4f}")
    print(f"Mean Predictive Entropy (Uncertainty): {final_metrics['UncertaintyEntropy']:.4f}")
    
    print("\n--- Fairness / Subgroup Analysis ---")
    print(f"{'Subgroup':<15} | {'AUC':<10} | {'FPR':<10} | {'Count':<5}")
    print("-" * 50)
    for name, metrics in final_metrics['Subgroups'].items():
        auc_val = metrics['AUC']
        fpr_val = metrics['FPR']
        print(f"{name:<15} | {auc_val:.4f}     | {fpr_val:.4f}     | {metrics['Count']}")

    # Save JSON Report
    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.integer): return int(obj)
            if isinstance(obj, np.floating): return float(obj)
            if isinstance(obj, np.ndarray): return obj.tolist()
            return super(NumpyEncoder, self).default(obj)

    with open(f'icare_project/model/cnn_bigru_model/{TRIAL_NAME}_evaluation_report.json', 'w') as f:
        json.dump(final_metrics, f, cls=NumpyEncoder, indent=4)
        
    print(f"\nReport saved to icare_project/model/cnn_bigru_model/{TRIAL_NAME}_evaluation_report.json")

if __name__ == "__main__":
    main()
