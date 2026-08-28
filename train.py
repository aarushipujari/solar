"""
☀️ Aditya-L1 Multi-Task Deep Learning Training Pipeline
Trains:
  1. 4-Channel Spatial Feature Encoder (UV + Sobel + Laplacian + Temporal Differential)
  2. Recurrent ConvLSTM Spatio-Temporal Sequence Cell
  3. Multi-Task Loss:
     - Loss_A: Binary M/X-Class Eruption (CrossEntropy)
     - Loss_B: NOAA 4-Class Flare Category [Quiet, C, M, X] (CrossEntropy)
     - Loss_C: Continuous Log10 Peak Flux Regression (MSE)
  4. Chronological Validation with True Skill Statistic (TSS) & F1 Tracking
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import BASE_DIR, DATA_DIR, BATCH_SIZE, SEQ_LENGTH, NUM_EPOCHS, LEARNING_RATE
from dataset import create_chronological_splits
from model import SolarFlarePredictor
from evaluate import evaluate_model_on_dataset


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing Spatio-Temporal Training on device: {device}")

    # 1. Load Chronological Dataset Splits (70% Train, 15% Val, 15% Test)
    train_ds, val_ds, test_ds = create_chronological_splits(DATA_DIR, seq_length=SEQ_LENGTH)
    print(f"Chronological Splits Loaded: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)} sequences.")

    if len(train_ds) == 0:
        print("Error: No training sequences found. Run 'python generate_sample_data.py' first.")
        return

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    # 2. Multi-Task Model & Objectives
    model = SolarFlarePredictor(in_channels=4, hidden_dim=32).to(device)

    criterion_binary = nn.CrossEntropyLoss()
    criterion_multiclass = nn.CrossEntropyLoss()
    criterion_flux = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)

    best_tss = -1.0
    best_weights_path = BASE_DIR / "solar_flare_model.pth"

    # 3. Training Loop
    print("\nStarting Multi-Task Spatio-Temporal Training Loop...")
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

            # Combined multi-task loss
            loss = loss_a + (0.5 * loss_b) + (0.2 * loss_c)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)

        # Validation Checkpoint
        val_eval = evaluate_model_on_dataset(model, val_ds if len(val_ds) > 0 else train_ds, device=device)
        bin_eval = val_eval["binary_evaluation"]

        print(
            f"Epoch [{epoch + 1:02d}/{NUM_EPOCHS}] "
            f"Loss: {avg_train_loss:.4f} | "
            f"Val Recall: {bin_eval['recall_tpr']:.2f} | "
            f"Val Precision: {bin_eval['precision']:.2f} | "
            f"Val F1: {bin_eval['f1_score']:.2f} | "
            f"Val TSS: {bin_eval['true_skill_statistic_tss']:.2f}"
        )

        if bin_eval["true_skill_statistic_tss"] >= best_tss:
            best_tss = bin_eval["true_skill_statistic_tss"]
            torch.save(model.state_dict(), best_weights_path)

    # 4. Final Evaluation on Held-Out Test Set
    print("\n========================================================")
    print("FINAL HELD-OUT CHRONOLOGICAL TEST SET EVALUATION")
    print("========================================================")
    model.load_state_dict(torch.load(best_weights_path, map_location=device))
    test_eval = evaluate_model_on_dataset(model, test_ds if len(test_ds) > 0 else train_ds, device=device)
    te_bin = test_eval["binary_evaluation"]

    print(f"Total Test Sequences: {test_eval['total_test_sequences']}")
    print(f"Recall (Sensitivity): {te_bin['recall_tpr'] * 100:.1f}%")
    print(f"Precision:            {te_bin['precision'] * 100:.1f}%")
    print(f"Specificity:          {te_bin['specificity'] * 100:.1f}%")
    print(f"F1-Score:             {te_bin['f1_score']:.4f}")
    print(f"True Skill Stat (TSS):{te_bin['true_skill_statistic_tss']:.4f}")
    print(f"Heidke Score (HSS):   {te_bin['heidke_skill_score_hss']:.4f}")
    print(f"ROC-AUC:              {te_bin['roc_auc']:.4f}")
    print(f"Log Flux MAE:         {test_eval['flux_mae_log10']:.4f}")
    print(f"Model saved to: {best_weights_path}")
    print("========================================================\n")


if __name__ == "__main__":
    run_training()