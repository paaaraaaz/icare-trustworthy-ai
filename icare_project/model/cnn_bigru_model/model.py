
import torch
import torch.nn as nn
import torch.nn.functional as F

class BCIEncoder(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64):
        super(BCIEncoder, self).__init__()
        # Input: (Batch, 1, 25920)
        # Reduce length 25920 -> manageable size
        
        # Conv Block 1: 25920 -> ~5184 (stride 5)
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=10, stride=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        
        # Conv Block 2: 5184 -> ~1296 (stride 4)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        # Conv Block 3: 1296 -> ~259 (stride 5)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Recurrent Layer to capture temporal dynamics of the reduced sequence
        self.gru = nn.GRU(128, hidden_dim, batch_first=True, bidirectional=True)
        # Output (Batch, Seq, 2*Hidden)
        
        self.global_pool = nn.AdaptiveAvgPool1d(1) 
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, x):
        # x: (Batch, SeqLen, 1) -> (Batch, 1, SeqLen)
        x = x.permute(0, 2, 1)
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.dropout(x, p=0.2, training=True) # Apply Monte Carlo Dropout for uncertainty estimation.
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.dropout(x, p=0.2, training=True)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.dropout(x, p=0.2, training=True)
        
        # Prepare for GRU: (Batch, Channels, Seq) -> (Batch, Seq, Channels)
        x = x.permute(0, 2, 1)
        
        out, _ = self.gru(x)
        # The output dimensions represent: (Batch, Sequence Length, 2 * Hidden Dimensions)
        
        # Global Pooling over time
        x = out.permute(0, 2, 1) # (Batch, Channels, Seq)
        x = self.global_pool(x).squeeze(-1)
        
        x = F.relu(self.fc(x))
        return x

class DemographicNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32):
        super(DemographicNet, self).__init__()
        # Structured clinical inputs: [Age, Sex, ROSC, OHCA, Shockable, TTM]
        self.fc1 = nn.Linear(input_dim, 32)
        self.fc2 = nn.Linear(32, hidden_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.1, training=True)
        x = F.relu(self.fc2(x))
        return x

class ICAREModel(nn.Module):
    def __init__(self, meta_dim=6):
        super(ICAREModel, self).__init__()
        self.branch_a = BCIEncoder(hidden_dim=64)
        self.branch_b = DemographicNet(input_dim=meta_dim, hidden_dim=32)
        
        self.fusion = nn.Linear(64 + 32, 32)
        self.output = nn.Linear(32, 1)
        
    def forward(self, x_bci, x_meta):
        # Branch A
        feat_a = self.branch_a(x_bci)
        
        # Branch B
        feat_b = self.branch_b(x_meta)
        
        # Fusion
        combined = torch.cat([feat_a, feat_b], dim=1)
        
        x = F.relu(self.fusion(combined))
        x = F.dropout(x, p=0.2, training=True) # Apply Monte Carlo Dropout at the fusion layer.
        
        logits = self.output(x)
        return logits # Return logits for BCEWithLogitsLoss

if __name__ == "__main__":
    # Test Model
    model = ICAREModel(meta_dim=6)
    print(model)
    
    # Dummy Input
    bci = torch.randn(2, 25920, 1)
    meta = torch.randn(2, 6)
    
    out = model(bci, meta)
    print("Output shape:", out.shape)
