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


class SpatioTemporalGradCAM:
    """
    Computes genuine Gradient-weighted Class Activation Mapping (Grad-CAM)
    across the temporal sequence of the CNN-ConvLSTM model using PyTorch autograd hooks.
    
    Formula:
      \\alpha_k^{(t)} = \\frac{1}{Z} \\sum_{i} \\sum_{j} \\frac{\\partial y^c}{\\partial A_{k,i,j}^{(t)}}
      L_{\\text{Grad-CAM}}^{(t)} = \\text{ReLU}\\left(\\sum_{k} \\alpha_k^{(t)} A_{k}^{(t)}\\right)
    """

    def __init__(self, model, target_layer=None):
        self.model = model
        # Target the final Conv2d layer in spatial encoder: model.encoder[3]
        self.target_layer = target_layer if target_layer is not None else model.encoder[3]
        self.activations = []
        self.gradients = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, out):
            self.activations.append(out)

        def backward_hook(module, grad_in, grad_out):
            # Prepend because backward runs in reverse order
            self.gradients.insert(0, grad_out[0])

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, seq_tensor, target_class=1):
        """
        Generates frame-by-frame Grad-CAM heatmaps for the given input sequence tensor.
        seq_tensor: [1, T, C, H, W]
        Returns: (list of 2D numpy arrays of shape [H, W], logits numpy array)
        """
        import cv2
        import numpy as np

        self.activations = []
        self.gradients = []
        self.model.eval()

        # Ensure gradient tracking
        input_var = seq_tensor.clone().detach().requires_grad_(True)
        logits = self.model(input_var)

        self.model.zero_grad()
        score = logits[0, target_class]
        score.backward()

        cams = []
        t = seq_tensor.shape[1]
        h_orig, w_orig = seq_tensor.shape[3], seq_tensor.shape[4]

        for i in range(min(t, len(self.activations), len(self.gradients))):
            act = self.activations[i]   # [1, 32, H', W']
            grad = self.gradients[i]    # [1, 32, H', W']

            # Global average pooling of gradients
            weights = torch.mean(grad, dim=(2, 3), keepdim=True)

            # Weighted linear combination
            cam = torch.sum(weights * act, dim=1, keepdim=True)
            cam = torch.relu(cam)

            cam_np = cam.squeeze().detach().cpu().numpy()

            # Normalize to [0, 1]
            denom = cam_np.max() - cam_np.min()
            if denom > 1e-8:
                cam_np = (cam_np - cam_np.min()) / denom
            else:
                cam_np = np.zeros_like(cam_np)

            # Resize to input patch resolution
            cam_resized = cv2.resize(cam_np, (w_orig, h_orig), interpolation=cv2.INTER_LINEAR)
            cams.append(cam_resized)

        return cams, logits.detach().cpu().numpy()[0]