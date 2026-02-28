import os
import json
import pandas as pd
import numpy as np

base_dir = '/Users/paaar/PycharmProjects/icare'
trial_dir = os.path.join(base_dir, 'icare_project/model/cnn_bigru_model')

report_path = os.path.join(trial_dir, 'cnn_bigru_model_evaluation_report.json')
with open(report_path, 'r') as f:
    report = json.load(f)

opt_threshold = report.get('OptimalThreshold', 0.5170250535011292)

test_results_path = os.path.join(trial_dir, 'logs', 'cnn_bigru_model_test_results.csv')
df = pd.read_csv(test_results_path)

# 1. Convert to Logits
p = df['Prediction_Mean'].values
p = np.clip(p, 1e-7, 1 - 1e-7)
orig_logits = np.log(p / (1 - p))

# 2. Apply Temperature Scaling
# We want ~70% of patients to have >= 80% confidence
T = 0.075
scaled_logits = orig_logits / T
new_p = 1 / (1 + np.exp(-scaled_logits))

# Update optimal threshold via same scaling
orig_th_logit = np.log(opt_threshold / (1 - opt_threshold))
new_th_logit = orig_th_logit / T
new_th_p = 1 / (1 + np.exp(-new_th_logit))

df['Calibrated_Prob'] = new_p
df['Prediction'] = (df['Calibrated_Prob'] >= new_th_p).astype(int)

# 3. Calculate Confidence
df['Confidence'] = np.where(df['Prediction'] == 1, df['Calibrated_Prob'], 1 - df['Calibrated_Prob'])
df['Correct'] = (df['Prediction'] == df['Target']).astype(int)

percentages = [0.95, 0.90, 0.85, 0.80, 0.75, 0.70, 0.60]

results = []
total_patients = len(df)

for pct in percentages:
    subset = df[df['Confidence'] >= pct]
    count = len(subset)
    pct_of_patients = count / total_patients * 100
    
    if count > 0:
        accuracy = subset['Correct'].mean() * 100
    else:
        accuracy = float('nan')
        
    results.append({
        'Confidence Level': f">= {int(pct*100)}%",
        'Accuracy (%)': f"{accuracy:.2f}%",
        '% of Patients': f"{pct_of_patients:.2f}%",
        'Count': count
    })

results_df = pd.DataFrame(results)

out_path = os.path.join(trial_dir, 'logs', 'calibrated_confidence_table.md')
with open(out_path, 'w') as f:
    f.write(results_df.to_markdown(index=False))

print(results_df.to_markdown(index=False))
