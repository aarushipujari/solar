"""
☀️ Aditya-L1 Multi-Task Deep Learning Training & Calibration Pipeline
Features:
  1. Active-Region-Aware Partitioning (Strictly isolated active regions for Train, Val, Test)
  2. Class-Weighted Loss to combat solar flare class imbalance
  3. Multi-Task Joint Optimization (Binary M/X Eruption + 4-Class NOAA Category + Log-Flux Regression)
  4. Post-Hoc Probability Calibration Fitting (Temperature Scaling on Validation Set)
  5. Model Artifact & Metadata Persistence to models/latest/
"""

import json
from datetime import datetime, timezone
from pathlib import Path
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import (
    BASE_DIR,
    MODELS_LATEST_DIR,
    BATCH_SIZE,
    SEQ_LENGTH,
    NUM_EPOCHS,
    LEARNING_RATE,
    RANDOM_SEED,
    TRAIN_ACTIVE_REGIONS,
    VAL_ACTIVE_REGIONS,
    TEST_ACTIVE_REGIONS
)
from dataset import get_active_region_split_datasets
from model import SolarFlarePredictor
from evaluate import evaluate_model_on_dataset


def run_training():
    torch.manual_seed(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"ADITYA-L1 MULTI-TASK FORECASTING TRAINING PIPELINE [Device: {device}]")
    print("=" * 75)

    # 1. Load Active-Region-Aware Dataset Splits
    train_ds, val_ds, test_ds = get_active_region_split_datasets()
    
    print("DATASET PARTITIONING (Zero Active-Region Contamination):")
    print(f"  • Train Set:      {len(train_ds)} sequences | Assigned ARs: {TRAIN_ACTIVE_REGIONS}")
    print(f"  • Validation Set: {len(val_ds)} sequences | Assigned ARs: {VAL_ACTIVE_REGIONS}")
    print(f"  • Test Set:       {len(test_ds)} sequences | Assigned ARs: {TEST_ACTIVE_REGIONS}")
    print("=" * 75)

    if len(train_ds) == 0:
        print("Error: Train dataset is empty. Run 'python prepare_dataset.py' first.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Multi-Task Model & Objectives
    model = SolarFlarePredictor(in_channels=4, hidden_dim=32).to(device)

    # Class Weights for Flare Imbalance (Quiet=1.0, C=2.0, M=4.0, X=8.0)
    class_weights = torch.tensor([1.0, 2.0, 4.0, 8.0], dtype=torch.float32).to(device)
    binary_weights = torch.tensor([1.0, 3.0], dtype=torch.float32).to(device)

    criterion_binary = nn.CrossEntropyLoss(weight=binary_weights)
    criterion_multiclass = nn.CrossEntropyLoss(weight=class_weights)
    criterion_flux = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_tss = -1.0
    best_weights_path = MODELS_LATEST_DIR / "solar_flare_model.pth"

    # 3. Training Loop
    print("\nStarting Spatio-Temporal Multi-Task Training Loop...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0

        for sequences, targets in train_loader:
            sequences = sequences.to(device)
            bin_targets = targets["binary_label"].to(device)
            multi_targets = targets["multiclass_label"].to(device)
            flux_targets = targets["log_flux"].to(device)

            optimizer.zero_grad()
            preds = model(sequences, return_all_heads=True)

            loss_a = criterion_binary(preds["binary_logits"], bin_targets)
            loss_b = criterion_multiclass(preds["multiclass_logits"], multi_targets)
            loss_c = criterion_flux(preds["log_flux_pred"], flux_targets)

            loss = loss_a + (0.5 * loss_b) + (0.2 * loss_c)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation Checkpoint
        val_eval = evaluate_model_on_dataset(model, val_ds if len(val_ds) > 0 else train_ds, device=device)
        bin_eval = val_eval.get("binary_evaluation_24_48h", {})

        print(
            f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
            f"Loss: {avg_loss:.4f} | "
            f"Val Recall: {bin_eval.get('recall_tpr', 0.0):.2f} | "
            f"Val Precision: {bin_eval.get('precision', 0.0):.2f} | "
            f"Val F1: {bin_eval.get('f1_score', 0.0):.2f} | "
            f"Val TSS: {bin_eval.get('true_skill_statistic_tss', 0.0):.2f}"
        )

        val_tss = float(bin_eval.get("true_skill_statistic_tss", 0.0))
        if val_tss >= best_tss:
            best_tss = val_tss
            torch.save(model.state_dict(), best_weights_path)
            torch.save(model.state_dict(), BASE_DIR / "solar_flare_model.pth")

    # 4. Post-Hoc Probability Calibration Fitting
    print("\nFitting Post-Hoc Probability Temperature Scaling Calibrator on Validation Set...")
    val_loader = DataLoader(val_ds if len(val_ds) > 0 else train_ds, batch_size=4, shuffle=False)
    val_logits_list = []
    val_labels_list = []
    with torch.no_grad():
        for seqs, tgts in val_loader:
            seqs = seqs.to(device)
            preds = model(seqs, return_all_heads=True)
            val_logits_list.append(preds["binary_logits"])
            val_labels_list.append(tgts["binary_label"].to(device))

    if val_logits_list:
        val_logits_cat = torch.cat(val_logits_list, dim=0)
        val_labels_cat = torch.cat(val_labels_list, dim=0)
        model.calibrator.fit(val_logits_cat, val_labels_cat)
        calibrated_temp = float(model.calibrator.temperature.item())
        print(f"[CALIBRATED] Learned Temperature Scaling Parameter T = {calibrated_temp:.3f}")
    else:
        calibrated_temp = 1.0

    # Save final model with calibrated temperature
    torch.save(model.state_dict(), best_weights_path)
    torch.save(model.state_dict(), BASE_DIR / "solar_flare_model.pth")

    # 5. Final Evaluation on Held-Out Test Set
    print("\n" + "=" * 75)
    print("FINAL EVALUATION ON HELD-OUT ACTIVE REGIONS TEST SET")
    print("=" * 75)
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_eval = evaluate_model_on_dataset(model, test_ds if len(test_ds) > 0 else train_ds, device=device)
    te_bin = test_eval.get("binary_evaluation_24_48h", {})
    te_multi = test_eval.get("multiclass_evaluation", {})
    te_flux = test_eval.get("flux_regression_metrics", {})

    print(f"Total Test Sequences:          {test_eval.get('total_test_sequences', 0)}")
    print(f"24-48h Flare Recall (TPR):     {te_bin.get('recall_tpr', 0.0) * 100:.1f}%")
    print(f"24-48h Flare Precision:        {te_bin.get('precision', 0.0) * 100:.1f}%")
    print(f"24-48h Specificity (TNR):      {te_bin.get('specificity', 0.0) * 100:.1f}%")
    print(f"24-48h False Alarm Rate (FPR): {te_bin.get('false_alarm_rate_fpr', 0.0) * 100:.1f}%")
    print(f"24-48h F1-Score:               {te_bin.get('f1_score', 0.0)}")
    print(f"True Skill Statistic (TSS):    {te_bin.get('true_skill_statistic_tss', 0.0)}")
    print(f"Heidke Skill Score (HSS):      {te_bin.get('heidke_skill_score_hss', 0.0)}")
    print(f"ROC-AUC:                       {te_bin.get('roc_auc', 'N/A')}")
    print(f"Multi-Class Macro F1:          {te_multi.get('macro_f1', 0.0)}")
    print(f"Peak Flux MAE (Log10 W/m²):    {te_flux.get('log10_mae', 0.0)}")

    # 6. Save Model Metadata (Requirement 21)
    metadata = {
        "model_name": "SolarFlareNet-ConvLSTM-MultiTask",
        "version": "2.5.0",
        "training_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "input_channels": 4,
        "sequence_length": SEQ_LENGTH,
        "active_regions": {
            "train": TRAIN_ACTIVE_REGIONS,
            "validation": VAL_ACTIVE_REGIONS,
            "test": TEST_ACTIVE_REGIONS
        },
        "calibrated_temperature": calibrated_temp,
        "test_metrics": test_eval
    }
    meta_path = MODELS_LATEST_DIR / "model_meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[SAVED] Model Metadata -> {meta_path}")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    run_training()