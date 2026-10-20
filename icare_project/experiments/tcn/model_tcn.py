import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# Temporal Convolutional Network (TCN)
# ==============================================================================
# An alternative to the CNN-BiGRU model.
# Uses dilated convolutions to see backward in time without using recurrent layers.
# ==============================================================================

class CausalConv1d(nn.Module):
    """
    1D convolution that only looks into the past.
    We add padding to the left side and cut off the right side
    so the model never peeks into future time points.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        out = self.conv(x)
        # Trim off the right padding to keep sequence length the same
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TemporalBlock(nn.Module):
    """
    A single TCN block with two causal convolutions, batch normalization,
    dropout, and a skip connection (residual add).
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.15):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)

        # Skip connection to help gradients flow easily
        self.residual = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        residual = self.residual(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        return F.relu(out + residual)


class TCNEncoder(nn.Module):
    """
    Stacks 6 temporal blocks with exponentially increasing dilation (1, 2, 4, 8, 16, 32).
    This allows the network to cover about 42 minutes of context per patient.
    """
    def __init__(self, input_channels=1, embed_dim=64, kernel_size=3, dropout=0.15):
        super().__init__()
        channels = [32, 32, 32, 64, 64, 64]
        dilations = [1, 2, 4, 8, 16, 32]

        layers = []
        in_ch = input_channels
        for ch, d in zip(channels, dilations):
            layers.append(TemporalBlock(in_ch, ch, kernel_size, d, dropout))
            in_ch = ch

        self.network = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(channels[-1], embed_dim)

    def forward(self, x):
        # Input shape: (Batch, 25920, 1) -> (Batch, 1, 25920)
        x = x.permute(0, 2, 1)
        x = self.network(x)
        x = self.global_pool(x).squeeze(-1)
        x = F.relu(self.fc(x))
        return x


class DemographicNet(nn.Module):
    """Small MLP to encode the 6 static clinical variables."""
    def __init__(self, input_dim=6, hidden_dim=32):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(32, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x


class ICAREModelTCN(nn.Module):
    """Combines the TCN sequence embedding with clinical features."""
    def __init__(self, meta_dim=6, embed_dim=64, kernel_size=3, dropout=0.15, fusion_dropout=0.2):
        super().__init__()
        self.branch_a = TCNEncoder(
            input_channels=1, embed_dim=embed_dim,
            kernel_size=kernel_size, dropout=dropout
        )
        self.branch_b = DemographicNet(input_dim=meta_dim, hidden_dim=32)

        self.fusion = nn.Linear(embed_dim + 32, 32)
        self.fusion_dropout = nn.Dropout(p=fusion_dropout)
        self.output = nn.Linear(32, 1)

    def forward(self, x_bci, x_meta):
        feat_a = self.branch_a(x_bci)
        feat_b = self.branch_b(x_meta)

        combined = torch.cat([feat_a, feat_b], dim=1)
        x = F.relu(self.fusion(combined))
        x = self.fusion_dropout(x)
        logits = self.output(x)
        return logits
