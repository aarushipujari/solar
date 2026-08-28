import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import BASE_DIR, DATA_DIR, BATCH_SIZE, SEQ_LENGTH, NUM_EPOCHS, LEARNING_RATE
from dataset import SolarSequenceDataset
from model import SolarFlarePredictor


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load dataset
    dataset = SolarSequenceDataset(data_dir=DATA_DIR, seq_length=SEQ_LENGTH)
    if len(dataset) == 0:
        print(f"Not enough contiguous FITS frames found in '{DATA_DIR}' to build sequence batches.")
        print("Run 'python generate_sample_data.py' to create sample FITS data, or set your data path in config.py.")
        return

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model and optimizer
    model = SolarFlarePredictor().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training Loop
    model.train()
    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        for sequences, labels in dataloader:
            sequences = sequences.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            predictions = model(sequences)
            loss = criterion(predictions, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}] - Loss: {total_loss / len(dataloader):.4f}")

    # Save trained model weights
    save_path = BASE_DIR / "solar_flare_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved successfully to {save_path}")


if __name__ == "__main__":
    run_training()