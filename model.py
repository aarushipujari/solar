"""
☀️ Aditya-L1 Spatio-Temporal Multi-Task Deep Learning Forecasting Architecture
Combines:
  1. 4-Channel Multi-Spectral & Topological Spatial Feature Encoder
  2. Recurrent ConvLSTM Spatio-Temporal Sequence Cell
  3. Multi-Task Heads:
     - Head A: 24-48h Binary M/X-Class Eruption Probability (Cross-Entropy)
     - Head B: 4-Class NOAA Flare Classification [Quiet/B, C-Class, M-Class, X-Class] (Cross-Entropy)
     - Head C: Continuous Log10 Peak X-Ray Flux Estimation (MSE Regression)
  4. Authentic PyTorch Autograd Spatio-Temporal Grad-CAM for Model Explainability
"""

import torch
import torch.nn as nn
import cv2
import numpy as np


class ConvLSTMCell(nn.Module):
    """Custom ConvLSTM block for extracting spatio-temporal feature evolutions."""

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
    """
    Multi-Channel Spatio-Temporal Multi-Task Forecasting Network.
    Input Shape: [Batch, Time (4), Channels (4), Height (256), Width (256)]
    Channels:
      - Ch 0: Calibrated UV / Intensity Patch
      - Ch 1: Spatial Flux Gradient (|∇I|)
      - Ch 2: High-Frequency Laplacian Curvature (∇²I)
      - Ch 3: Temporal Flux Evolution (ΔI_t)
    """

    def __init__(self, in_channels=4, hidden_dim=32):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # 2D Multi-Channel Spatial Feature Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # [B, 16, 128, 128]
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),            # [B, 32, 64, 64]
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        # Spatio-Temporal Sequence Cell
        self.conv_lstm = ConvLSTMCell(in_channels=32, hidden_channels=hidden_dim)

        # Pooling Layer
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        # Multi-Task Prediction Heads
        # Head A: 24-48h Binary M/X-Class Flare Eruption (0: Low/No Flare, 1: M/X Flare)
        self.binary_head = nn.Linear(hidden_dim, 2)

        # Head B: NOAA Multi-Class Flare Classification [0: Quiet/B, 1: C-Class, 2: M-Class, 3: X-Class]
        self.multiclass_head = nn.Linear(hidden_dim, 4)

        # Head C: Continuous Log10 Peak X-Ray Flux (e.g. -7.5 W/m² to -3.5 W/m²)
        self.flux_regression_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_all_heads=True):
        """
        Forward pass.
        If return_all_heads is True: returns dict of {'binary': ..., 'multiclass': ..., 'log_flux': ...}
        If return_all_heads is False: returns binary logits for backwards compatibility.
        """
        b, t, c, h, w = x.shape

        # Support single-channel inputs by auto-expanding if needed
        if c == 1 and self.in_channels == 4:
            x = x.repeat(1, 1, 4, 1, 1)
            c = 4

        h_state = torch.zeros(b, self.hidden_dim, h // 4, w // 4, device=x.device)
        c_state = torch.zeros(b, self.hidden_dim, h // 4, w // 4, device=x.device)

        # Recurrent sequence unrolling
        for i in range(t):
            frame_input = x[:, i, :, :, :]
            spatial_features = self.encoder(frame_input)
            h_state, c_state = self.conv_lstm(spatial_features, h_state, c_state)

        # Bottleneck feature vector
        features = self.pool(h_state).flatten(start_dim=1)  # [B, hidden_dim]

        binary_logits = self.binary_head(features)
        multiclass_logits = self.multiclass_head(features)
        log_flux_pred = self.flux_regression_head(features).squeeze(-1)

        if not return_all_heads:
            return binary_logits

        return {
            "binary_logits": binary_logits,
            "multiclass_logits": multiclass_logits,
            "log_flux_pred": log_flux_pred
        }


class SpatioTemporalGradCAM:
    """
    Computes genuine Gradient-weighted Class Activation Mapping (Grad-CAM)
    across the temporal sequence of the CNN-ConvLSTM model using PyTorch autograd hooks.
    
    Formula:
      α_k^(t) = (1/Z) Σ_{i,j} (∂y^c / ∂A_{k,i,j}^(t))
      L_{Grad-CAM}^(t) = ReLU(Σ_k α_k^(t) A_k^(t))
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        self.target_layer = target_layer if target_layer is not None else model.encoder[3]
        self.activations = []
        self.gradients = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            self.gradients.insert(0, grad_out[0])

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, seq_tensor, target_class=1, task="binary"):
        """
        Generates frame-by-frame Grad-CAM heatmaps for the given input sequence tensor.
        seq_tensor: [1, T, C, H, W]
        Returns: (list of 2D numpy arrays of shape [H, W], predictions_dict)
        """
        self.activations = []
        self.gradients = []
        self.model.eval()

        input_var = seq_tensor.clone().detach().requires_grad_(True)
        preds = self.model(input_var, return_all_heads=True)

        self.model.zero_grad()
        if task == "binary":
            score = preds["binary_logits"][0, target_class]
        else:
            score = preds["multiclass_logits"][0, target_class]

        score.backward()

        cams = []
        t = seq_tensor.shape[1]
        h_orig, w_orig = seq_tensor.shape[3], seq_tensor.shape[4]

        for i in range(min(t, len(self.activations), len(self.gradients))):
            act = self.activations[i]
            grad = self.gradients[i]

            weights = torch.mean(grad, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * act, dim=1, keepdim=True)
            cam = torch.relu(cam)

            cam_np = cam.squeeze().detach().cpu().numpy()

            denom = cam_np.max() - cam_np.min()
            if denom > 1e-8:
                cam_np = (cam_np - cam_np.min()) / denom
            else:
                cam_np = np.zeros_like(cam_np)

            cam_resized = cv2.resize(cam_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            cams.append(cam_resized)

        # Convert predictions to numpy
        clean_preds = {
            "binary_probs": torch.softmax(preds["binary_logits"], dim=1).detach().cpu().numpy()[0],
            "multiclass_probs": torch.softmax(preds["multiclass_logits"], dim=1).detach().cpu().numpy()[0],
            "log_flux_pred": float(preds["log_flux_pred"].detach().cpu().numpy()[0])
        }

        return cams, clean_preds