import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

base_dir = '/Users/paaar/PycharmProjects/icare'
trial_dir = os.path.join(base_dir, 'icare_project/model/cnn_bigru_model')
out_dir = os.path.join(trial_dir, 'visuals')
os.makedirs(out_dir, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.5)
blue_color = '#1f77b4'
orange_color = '#ff7f0e'

# Load trial data
report_path = os.path.join(trial_dir, 'cnn_bigru_model_evaluation_report.json')
with open(report_path, 'r') as f:
    report = json.load(f)

test_results_path = os.path.join(trial_dir, 'logs', 'cnn_bigru_model_test_results.csv')
df = pd.read_csv(test_results_path)

opt_threshold = report.get('OptimalThreshold', 0.5)

def plot_metrics_and_cm():
    # 1. Classification Metrics @ Optimal Threshold
    y_true = df['Target']
    y_pred = (df['Prediction_Mean'] >= opt_threshold).astype(int)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    # Save Metrics Plot
    metrics = ['Accuracy', 'Precision (PPV)', 'Recall (Sensitivity)', 'F1-Score']
    values = [acc, prec, rec, f1]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(metrics, values, color=['#2ca02c', '#d62728', '#9467bd', '#8c564b'])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel('Score')
    ax.set_title(f'Trial 6: Classification Metrics (Threshold = {opt_threshold:.3f})')
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f"{yval:.4f}", ha='center', va='bottom', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig3_classification_metrics.png'), dpi=300)
    plt.close()

    # 2. Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, annot_kws={"size": 16}, cbar=False)
    ax.set_xlabel('Predicted Label (Outcome)', fontsize=14)
    ax.set_ylabel('True Label (Outcome)', fontsize=14)
    ax.set_title(f'Trial 6: Confusion Matrix (Threshold = {opt_threshold:.3f})', fontsize=16)
    ax.xaxis.set_ticklabels(['Good Outcome (0)', 'Poor Outcome (1)'])
    ax.yaxis.set_ticklabels(['Good Outcome (0)', 'Poor Outcome (1)'])
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig4_confusion_matrix.png'), dpi=300)
    plt.close()
    
    # Check parity on precision and recall before writing the report
    return acc, prec, rec, f1

def plot_auc_subgroups():
    # 3. Fairness Audit: AUC by Subgroup
    subgroups = report['Subgroups']
    sex_labels = ['Male', 'Female']
    age_labels = ['Age < 60', 'Age >= 60']
    
    sex_aucs = [subgroups[k]['AUC'] for k in sex_labels]
    age_aucs = [subgroups[k]['AUC'] for k in age_labels]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    bar_width = 0.35
    index = np.arange(2)
    
    rects1 = ax.bar(index, sex_aucs, bar_width, label='Sex', color=blue_color)
    rects2 = ax.bar(index + bar_width + 0.1, age_aucs, bar_width, label='Age', color=orange_color)
    
    ax.set_ylim(0.6, 0.9)
    ax.set_ylabel('Area Under ROC Curve (AUC)')
    ax.set_title('Trial 6: Model Fairness across Demographic Subgroups')
    ax.set_xticks(np.concatenate([index, index + bar_width + 0.1]))
    ax.set_xticklabels(sex_labels + age_labels)
    
    overall_auc = report['AUC']
    ax.axhline(overall_auc, color='red', linestyle='--', linewidth=2, label=f'Overall AUC ({overall_auc:.3f})')
    
    for rect in rects1 + rects2:
        height = rect.get_height()
        ax.annotate(f'{height:.3f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12)

    ax.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig1_fairness_auc.png'), dpi=300)
    plt.close()

def plot_uncertainty():
    # 4. Uncertainty (Entropy) Distribution Analysis
    df['Prediction'] = (df['Prediction_Mean'] >= opt_threshold).astype(int)
    
    df['Outcome_Type'] = 'Unknown'
    df.loc[(df['Target'] == 1) & (df['Prediction'] == 1), 'Outcome_Type'] = 'True Positive'
    df.loc[(df['Target'] == 0) & (df['Prediction'] == 0), 'Outcome_Type'] = 'True Negative'
    df.loc[(df['Target'] == 0) & (df['Prediction'] == 1), 'Outcome_Type'] = 'False Positive'
    df.loc[(df['Target'] == 1) & (df['Prediction'] == 0), 'Outcome_Type'] = 'False Negative'
    
    mean_entropy = df.groupby('Outcome_Type')['Uncertainty_Entropy'].mean()
    pred_types = ['True Positive', 'True Negative', 'False Positive', 'False Negative']
    
    # In case a category has no items
    entropies = []
    for p in pred_types:
        if p in mean_entropy.index:
            entropies.append(mean_entropy[p])
        else:
            entropies.append(0.0)
            
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = [blue_color, blue_color, '#d62728', '#d62728']
    bars = ax.bar(pred_types, entropies, color=colors)
    
    ax.set_ylabel('Mean Predictive Entropy')
    ax.set_title('Trial 6: Uncertainty Calibration by Prediction Outcome')
    ax.set_ylim(0.4, 0.75)
    
    overall_entropy = df['Uncertainty_Entropy'].mean()
    ax.axhline(overall_entropy, color='black', linestyle=':', label=f'Overall Mean ({overall_entropy:.3f})')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.005, f"{yval:.4f}", ha='center', va='bottom', fontsize=12)
        
    ax.legend(loc='upper right')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'fig2_uncertainty_entropy.png'), dpi=300)
    plt.close()

if __name__ == "__main__":
    acc, prec, rec, f1 = plot_metrics_and_cm()
    plot_auc_subgroups()
    plot_uncertainty()
    
    print(f"Metrics Output: Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
    print("Visualizations generated in icare_project/model/cnn_bigru_model/visuals/")
