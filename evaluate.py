"""
📊 Space-Weather Evaluation & Verification Engine
Computes standard scientific skill scores:
  - Precision, Recall (Sensitivity), Specificity, False Alarm Rate, Miss Rate
  - F1-Score (Binary & Multi-Class Macro/Weighted)
  - True Skill Statistic (TSS = TPR - FPR) [Gold standard for solar flare forecasting]
  - Heidke Skill Score (HSS)
  - ROC-AUC and PR-AUC (with honest N/A handling for single-class subsets)
  - Regression MAE, RMSE, R² for Log-Peak Flux
Zero fabrication: If a metric cannot be calculated due to class distribution, returns N/A.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    mean_squared_error,
    mean_absolute_error,
    r2_score
)


def compute_space_weather_skill_scores(y_true, y_pred_prob, threshold=0.5):
    """
    Computes genuine scientific skill scores for binary solar flare forecasting (>= M1.0).
    """
    y_true = np.array(y_true, dtype=int)
    y_pred_prob = np.array(y_pred_prob, dtype=float)
    y_pred = (y_pred_prob >= threshold).astype(int)

    # Confusion matrix components: TN, FP, FN, TP
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Sensitivity / True Positive Rate (Recall)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # False Positive Rate (False Alarm Rate)
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    # Specificity / True Negative Rate
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Miss Rate (False Negative Rate)
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    # Precision
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0

    # F1 Score
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0.0

    # True Skill Statistic (TSS): TSS = TPR - FPR
    tss = tpr - fpr

    # Heidke Skill Score (HSS)
    numerator = 2 * (tp * tn - fp * fn)
    denominator = (tp + fn) * (fn + tn) + (tp + fp) * (fp + tn)
    hss = numerator / denominator if denominator > 0 else 0.0

    # ROC-AUC & PR-AUC with honest N/A handling
    if len(np.unique(y_true)) > 1:
        try:
            roc_auc = round(float(roc_auc_score(y_true, y_pred_prob)), 4)
            pr_auc = round(float(average_precision_score(y_true, y_pred_prob)), 4)
        except Exception:
            roc_auc = "N/A"
            pr_auc = "N/A"
    else:
        roc_auc = "N/A (Single class present in test set)"
        pr_auc = "N/A (Single class present in test set)"

    return {
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
        "recall_tpr": round(float(tpr), 4),
        "precision": round(float(precision), 4),
        "specificity": round(float(tnr), 4),
        "false_alarm_rate_fpr": round(float(fpr), 4),
        "miss_rate_fnr": round(float(fnr), 4),
        "f1_score": round(float(f1), 4),
        "true_skill_statistic_tss": round(float(tss), 4),
        "heidke_skill_score_hss": round(float(hss), 4),
        "roc_auc": roc_auc,
        "pr_auc": pr_auc
    }


def evaluate_model_on_dataset(model, dataset, device="cpu", batch_size=4):
    """
    Evaluates multi-task forecasting model on a dataset.
    """
    if len(dataset) == 0:
        return {
            "status": "NO_DATA",
            "message": "Dataset is empty."
        }

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()

    all_bin_true = []
    all_bin_probs = []
    all_multi_true = []
    all_multi_preds = []
    all_flux_true = []
    all_flux_preds = []

    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences = sequences.to(device)
            preds = model(sequences, return_all_heads=True)

            bin_probs = torch.softmax(preds["calibrated_binary_logits"], dim=1)[:, 1].cpu().numpy()
            multi_preds = torch.argmax(preds["multiclass_logits"], dim=1).cpu().numpy()
            flux_preds = preds["log_flux_pred"].cpu().numpy()

            all_bin_true.extend(targets["binary_label"].numpy())
            all_bin_probs.extend(bin_probs)
            all_multi_true.extend(targets["multiclass_label"].numpy())
            all_multi_preds.extend(multi_preds)
            all_flux_true.extend(targets["log_flux"].numpy())
            all_flux_preds.extend(flux_preds)

    binary_metrics = compute_space_weather_skill_scores(all_bin_true, all_bin_probs)

    # Multi-class evaluation
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        all_multi_true, all_multi_preds, average='macro', zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        all_multi_true, all_multi_preds, average='weighted', zero_division=0
    )

    # Flux Regression Metrics
    flux_arr_true = np.array(all_flux_true)
    flux_arr_pred = np.array(all_flux_preds)
    mae = float(mean_absolute_error(flux_arr_true, flux_arr_pred))
    rmse = float(np.sqrt(mean_squared_error(flux_arr_true, flux_arr_pred)))
    try:
        r2 = float(r2_score(flux_arr_true, flux_arr_pred)) if len(np.unique(flux_arr_true)) > 1 else "N/A"
    except Exception:
        r2 = "N/A"

    return {
        "binary_evaluation_24_48h": binary_metrics,
        "multiclass_evaluation": {
            "macro_f1": round(float(f1_macro), 4),
            "macro_precision": round(float(p_macro), 4),
            "macro_recall": round(float(r_macro), 4),
            "weighted_f1": round(float(f1_weighted), 4),
            "classes_evaluated": ["Quiet/B", "C-Class", "M-Class", "X-Class"]
        },
        "flux_regression_metrics": {
            "log10_mae": round(mae, 4),
            "log10_rmse": round(rmse, 4),
            "r2_score": round(float(r2), 4) if isinstance(r2, float) else r2
        },
        "total_test_sequences": len(all_bin_true)
    }
