# src/model.py
import torch
import torch.nn as nn

def conv_block(in_channels, out_channels, kernel_size=3, dropout=0.0):
    padding = kernel_size // 2
    return nn.Sequential(
        nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
        nn.BatchNorm1d(out_channels),
        nn.ReLU(inplace=True),
        nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    )

class CNN1DRegressor(nn.Module):
    def __init__(self, input_channels=3, nlayers=3, dropout=0.2):
        super().__init__()
        layers = []
        out_ch = 32
        in_ch = input_channels
        for i in range(nlayers):
            layers.append(conv_block(in_ch, out_ch, kernel_size=3, dropout=dropout))
            in_ch = out_ch
            if i < nlayers - 1:
                out_ch *= 2
        self.conv_blocks = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(in_ch, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # (B, W, F) → (B, F, W)
        x = self.conv_blocks(x)
        x = self.global_pool(x).flatten(1)
        x = self.fc(x)
        return x.squeeze(-1)