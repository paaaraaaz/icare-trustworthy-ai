import torch
import torch.nn as nn
import torch.nn.functional as F

class BCIEncoder(nn.Module):
    def __init__(self, input_channels=1, hidden_dim=64, use_gru=True, use_cnn=True):
        super(BCIEncoder, self).__init__()
        self.use_gru = use_gru
        self.use_cnn = use_cnn
        
        if self.use_cnn:
            # 3-stage Conv1d feature extractor
            self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=10, stride=5, padding=2)
            self.bn1 = nn.BatchNorm1d(32)
            
            self.conv2 = nn.Conv1d(32, 64, kernel_size=8, stride=4, padding=2)
            self.bn2 = nn.BatchNorm1d(64)
            
            self.conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=5, padding=2)
            self.bn3 = nn.BatchNorm1d(128)
            
            self.dropout1 = nn.Dropout(p=0.2)
            self.dropout2 = nn.Dropout(p=0.2)
            self.dropout3 = nn.Dropout(p=0.2)
            gru_in_dim = 128
        else:
            # Pure signal pooling without convolutional learnable filters (GRU-only variant)
            self.pool_down = nn.AvgPool1d(kernel_size=100, stride=100)
            gru_in_dim = input_channels
        
        if self.use_gru:
            self.gru = nn.GRU(gru_in_dim, hidden_dim, batch_first=True, bidirectional=True)
            self.global_pool = nn.AdaptiveAvgPool1d(1) 
            self.fc = nn.Linear(hidden_dim * 2, hidden_dim)
        else:
            self.global_pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Linear(128, hidden_dim)

    def forward(self, x):
        # Ensure tensor is formatted as (Batch, Channels, Length) for Conv1d / AvgPool1d
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.permute(0, 2, 1)
        elif x.dim() == 2:
            x = x.unsqueeze(1)
            
        if self.use_cnn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = self.dropout1(x)
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.dropout2(x)
            x = F.relu(self.bn3(self.conv3(x)))
            x = self.dropout3(x)
            # (Batch, 128, Seq) -> (Batch, Seq, 128)
            feat = x.permute(0, 2, 1)
        else:
            # Non-convolutional downsampled sequence: (Batch, 1, 259) -> (Batch, 259, 1)
            x = self.pool_down(x)
            feat = x.permute(0, 2, 1)
        
        if self.use_gru:
            out, _ = self.gru(feat) # (Batch, Seq, hidden_dim * 2)
            x = out.permute(0, 2, 1) # (Batch, hidden_dim * 2, Seq)
            x = self.global_pool(x).squeeze(-1) # (Batch, hidden_dim * 2)
        else:
            x = feat.permute(0, 2, 1) # (Batch, 128, Seq)
            x = self.global_pool(x).squeeze(-1) # (Batch, 128)
            
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
    def __init__(self, meta_dim=6, use_gru=True, use_meta=True, use_cnn=True, input_channels=1):
        super(ICAREModelAblation, self).__init__()
        self.use_meta = use_meta
        self.use_gru = use_gru
        self.use_cnn = use_cnn
        
        self.branch_a = BCIEncoder(input_channels=input_channels, hidden_dim=64, use_gru=use_gru, use_cnn=use_cnn)
        
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

if __name__ == "__main__":
    bci = torch.randn(2, 25920, 1)
    meta = torch.randn(2, 6)
    
    variants = [
        ("Full", dict(use_gru=True, use_cnn=True, use_meta=True)),
        ("No-Clinical", dict(use_gru=True, use_cnn=True, use_meta=False)),
        ("CNN-only", dict(use_gru=False, use_cnn=True, use_meta=True)),
        ("GRU-only", dict(use_gru=True, use_cnn=False, use_meta=True)),
    ]
    for name, cfg in variants:
        m = ICAREModelAblation(**cfg)
        out = m(bci, meta)
        print(f"Variant: {name:12s} | Output: {out.shape} | Params: {sum(p.numel() for p in m.parameters()):,}")
