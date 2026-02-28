import torch
import torch.nn as nn
import torch.nn.functional as F

class BCIEncoder(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64, use_gru=True):
        super(BCIEncoder, self).__init__()
        self.use_gru = use_gru
        
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=10, stride=5, padding=2)
        self.bn1 = nn.BatchNorm1d(32)
        
        self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=5, padding=2)
        self.bn3 = nn.BatchNorm1d(128)
        
        # Instantiate explicit Dropout layers to enforce regularization.
        self.dropout1 = nn.Dropout(p=0.2)
        self.dropout2 = nn.Dropout(p=0.2)
        self.dropout3 = nn.Dropout(p=0.2)
        
        if self.use_gru:
            self.gru = nn.GRU(128, hidden_dim, batch_first=True, bidirectional=True)
            self.global_pool = nn.AdaptiveAvgPool1d(1) 
            self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(128, hidden_dim)

    def forward(self, x):
        # Ensure the input tensor x is formatted as (Batch Size, Channels, Sequence Length).
        
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.dropout1(x)
        
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.dropout2(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.dropout3(x)
        
        if self.use_gru:
            x = x.permute(0, 2, 1) # (Batch, Seq, Channels)
            out, _ = self.gru(x)
            x = out.permute(0, 2, 1) # (Batch, Channels, Seq)
            x = self.global_pool(x).squeeze(-1)
        else:
            x = self.global_pool(x).squeeze(-1)
            
        x = F.relu(self.fc(x))
        return x

class DemographicNet(nn.Module):
    def __init__(self, input_dim=6, hidden_dim=32):
        super(DemographicNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(32, hidden_dim)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        return x

class ICAREModelAblation(nn.Module):
    def __init__(self, meta_dim=6, use_gru=True, use_meta=True, input_channels=1):
        super(ICAREModelAblation, self).__init__()
        self.use_meta = use_meta
        self.use_gru = use_gru
        
        self.branch_a = BCIEncoder(input_channels=input_channels, hidden_dim=64, use_gru=use_gru)
        
        if self.use_meta:
            self.branch_b = DemographicNet(input_dim=meta_dim, hidden_dim=32)
            self.fusion = nn.Linear(64 + 32, 32)
        else:
            self.fusion = nn.Linear(64, 32)
            
        self.fusion_dropout = nn.Dropout(p=0.2)
        self.output = nn.Linear(32, 1)
        
    def forward(self, x_bci, x_meta):
        feat_a = self.branch_a(x_bci)
        
        if self.use_meta:
            feat_b = self.branch_b(x_meta)
            combined = torch.cat([feat_a, feat_b], dim=1)
        else:
            combined = feat_a
            
        x = F.relu(self.fusion(combined))
        x = self.fusion_dropout(x)
        
        logits = self.output(x)
        return logits
