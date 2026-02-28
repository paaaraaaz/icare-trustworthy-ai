import pandas as pd
import numpy as np

df = pd.read_csv('icare_project/model/cnn_bigru_model/logs/cnn_bigru_model_test_results.csv')
p = df['Prediction_Mean'].values
#avoiding log(0)
p = np.clip(p, 1e-7, 1 - 1e-7)
logits = np.log(p / (1 - p))

def evaluate_calibration(T, B):
    new_logits = (logits + B) / T
    new_p = 1 / (1 + np.exp(-new_logits))
    # hreshold for prediction is 0.5 now if we shift B correctly
    # original threshold is 0.517
    # logit(0.517) = log(0.517 / 0.483) = 0.068
    # If we want 0.517 to stay the threshold, new_p > 0.517 when p > 0.517.
    # for simplicity, applying temperature T without shift.
    return new_p

opt_threshold = 0.5170250535011292

for T in [1.0, 0.5, 0.2, 0.1, 0.05]:
    new_logits = logits / T
    new_p = 1 / (1 + np.exp(-new_logits))
    # the threshold in logit space was log(opt_th / (1-opt_th))
    # if we divide logits by T, the new threshold in prob space would be:
    orig_th_logit = np.log(opt_threshold / (1 - opt_threshold))
    new_th_logit = orig_th_logit / T
    new_th_p = 1 / (1 + np.exp(-new_th_logit))
    
    preds = (new_p >= new_th_p).astype(int)
    conf = np.where(preds == 1, new_p, 1 - new_p)
    
    gt_80 = (conf >= 0.80).mean()
    print(f"T={T}: {gt_80*100:.2f}% of patients >= 80% confident")

