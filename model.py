import torch
import torch.nn as nn


class ConvLSTMCell(nn.Module):
    """Custom ConvLSTM block for extracting spatio-temporal features."""

    def __init__(self, in_channels, hidden_channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.conv = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size,
            padding=padding
        )

    def forward(self, x, h, c):
        combined = torch.cat([x, h], dim=1)
        gates = self.conv(combined)
        ingate, forgetgate, cellgate, outgate = torch.split(gates, self.hidden_channels, dim=1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        new_c = (forgetgate * c) + (ingate * cellgate)
        new_h = outgate * torch.tanh(new_c)
        return new_h, new_c


class SolarFlarePredictor(nn.Module):
    """Complete Model: CNN Feature Extractor + ConvLSTM + Dense Classifier."""

    def __init__(self, hidden_dim=32):
        super().__init__()
        # 2D Spatial Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 128, 128]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),  # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv_lstm = ConvLSTMCell(in_channels=32, hidden_channels=hidden_dim)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(hidden_dim, 2)  # Output: [No Flare, Flare Eruption]
        )

    def forward(self, x):
        # Input shape: [Batch, Time, Channels, Height, Width]
        b, t, c, h, w = x.shape

        # Initialize LSTM hidden states
        h_state = torch.zeros(b, 32, h // 4, w // 4, device=x.device)
        c_state = torch.zeros(b, 32, h // 4, w // 4, device=x.device)

        # Process each frame in the sequence chronologically
        for i in range(t):
            spatial_features = self.encoder(x[:, i, :, :, :])
            h_state, c_state = self.conv_lstm(spatial_features, h_state, c_state)

        return self.classifier(h_state)