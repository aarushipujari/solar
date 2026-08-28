import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from config import DATA_DIR, BATCH_SIZE, SEQ_LENGTH, NUM_EPOCHS, LEARNING_RATE
from dataset import SolarSequenceDataset
from model import SolarFlarePredictor


def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    # Load dataset
    dataset = SolarSequenceDataset(data_dir=DATA_DIR, seq_length=SEQ_LENGTH)
    if len(dataset) == 0:
        print("Not enough contiguous FITS frames found to build sequence batches.")
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


if __name__ == "__main__":
    run_training()