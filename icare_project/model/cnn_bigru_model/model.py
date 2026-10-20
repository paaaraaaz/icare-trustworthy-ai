import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================================================================
# 1. BCI Temporal Encoder (CNN + BiGRU)
# ==============================================================================
# This module reads the 72-hour EEG Brain Continuity Index (BCI) time series.
# The raw input has 25,920 time points (one measurement every 10 seconds).
# We shrink this long signal with 3 convolutional layers so the GRU can handle it.
# ==============================================================================

class BCIEncoder(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64, dropout_rate=0.2):
        super(BCIEncoder, self).__init__()
        
        # Conv Block 1: Shrinks 25,920 points down to ~5,184 points (stride of 5)
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=10, stride=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        self.drop1 = nn.Dropout(p=dropout_rate)
        
        # Conv Block 2: Shrinks 5,184 points down to ~1,296 points (stride of 4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(p=dropout_rate)
        
        # Conv Block 3: Shrinks 1,296 points down to ~259 points (stride of 5)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        self.drop3 = nn.Dropout(p=dropout_rate)
        
        # Bidirectional GRU: Reads the compressed 259 time points forward and backward
        # to capture recovery patterns and sudden drops in brain continuity.
        self.gru = nn.GRU(128, hidden_dim, batch_first=True, bidirectional=True)
        
        # Pool across all time points to create a single summary vector
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Final linear layer to project the BiGRU summary
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        # Input shape: (Batch, 25920, 1) -> reorder to (Batch, 1, 25920) for 1D convolution
        x = x.permute(0, 2, 1)
        
        # Run through the 3 convolutional shrinking blocks
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.drop1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.drop2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.drop3(x)
        
        # Reorder back for GRU: (Batch, Channels, Time) -> (Batch, Time, Channels)
        x = x.permute(0, 2, 1)
        
        # Run the bidirectional GRU
        out, _ = self.gru(x)
        
        # Average across all time steps
        x = out.permute(0, 2, 1)
        x = self.global_pool(x).squeeze(-1)
        
        # Final activation
        x = F.relu(self.fc(x))
        return x


# ==============================================================================
# 2. Demographic and Clinical Feature Encoder
# ==============================================================================
# This small network processes the 6 static patient variables:
# [Age, Sex, ROSC duration, Out-of-Hospital status, Initial rhythm, TTM]
# ==============================================================================

class DemographicNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32, dropout_rate=0.1):
        super(DemographicNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.drop = nn.Dropout(p=dropout_rate)
        self.fc2 = nn.Linear(32, hidden_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = F.relu(self.fc2(x))
        return x


# ==============================================================================
# 3. Multimodal CNN-BiGRU Full Model
# ==============================================================================
# This model combines the EEG sequence embedding with the clinical embedding,
# passes them through a fusion head, and outputs the risk of poor outcome.
# ==============================================================================

class MultimodalICAREModel(nn.Module):
    def __init__(self, gru_hidden=64, dropout_rate=0.2, fusion_dropout=0.2):
        super(MultimodalICAREModel, self).__init__()
        self.bci_encoder = BCIEncoder(input_channels=1, hidden_dim=gru_hidden, dropout_rate=dropout_rate)
        self.demo_encoder = DemographicNet(input_dim=6, hidden_dim=32, dropout_rate=dropout_rate)
        
        # Fusion head: concatenates EEG features (gru_hidden) and clinical features (32)
        self.fusion = nn.Sequential(
            nn.Linear(gru_hidden + 32, 32),
            nn.ReLU(),
            nn.Dropout(p=fusion_dropout),
            nn.Linear(32, 1) # Outputs a single logit (poor outcome vs good outcome)
        )

    def forward(self, bci, demo):
        bci_feats = self.bci_encoder(bci)
        demo_feats = self.demo_encoder(demo)
        combined = torch.cat([bci_feats, demo_feats], dim=1)
        logits = self.fusion(combined)
        return logits
