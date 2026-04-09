# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
from torchvision.models import resnet18

class LipReadingModel(nn.Module):
    def __init__(self, num_classes=500, d_model=512, nhead=8,
                 num_layers=4, dropout=0.1):
        super().__init__()

        # ── Frontend: ResNet18 as spatial feature extractor ──
        resnet = resnet18(weights=None)
        # Modify first conv to accept 1 channel (grayscale)
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2,
                                  padding=3, bias=False)
        # Remove final FC layer — we just want features
        self.frontend = nn.Sequential(*list(resnet.children())[:-1])
        # Output: (B*T, 512, 1, 1)

        # ── Positional Encoding ──
        self.pos_encoding = PositionalEncoding(d_model, dropout)

        # ── Transformer Encoder ──
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer,
                                                  num_layers=num_layers)

        # ── Classifier ──
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, T, 1, H, W)
        B, T, C, H, W = x.shape

        # Extract per-frame features with ResNet
        x = x.view(B * T, C, H, W)           # (B*T, 1, H, W)
        x = self.frontend(x)                   # (B*T, 512, 1, 1)
        x = x.view(B, T, -1)                  # (B, T, 512)

        # Add positional encoding
        x = self.pos_encoding(x)              # (B, T, 512)

        # Transformer over time
        x = self.transformer(x)               # (B, T, 512)

        # Global average pool over time
        x = x.mean(dim=1)                     # (B, 512)

        # Classify
        x = self.classifier(x)                # (B, num_classes)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=500):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float()
                             * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)