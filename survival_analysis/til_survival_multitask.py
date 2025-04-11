# til_survival_multitask.py
# Transformer-based survival prediction model that handles TIL time-series data and outputs multiple competing risks.

import torch
import torch.nn as nn
import torch.nn.functional as F

class TILPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)].to(x.device)

class TILSurvivalMultiHead(nn.Module):
    def __init__(self, input_dim=64, model_dim=128, num_layers=2, nhead=4, num_outputs=3):
        """
        Initializes a survival model with competing risks output.

        Args:
            input_dim (int): Input feature size per timepoint (e.g. TIL subtype composition).
            model_dim (int): Internal transformer dimension.
            num_layers (int): Transformer encoder depth.
            nhead (int): Number of attention heads.
            num_outputs (int): Number of competing risks to predict.
        """
        super().__init__()
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.pos_encoder = TILPositionalEncoding(model_dim)
        encoder_layer = nn.TransformerEncoderLayer(d_model=model_dim, nhead=nhead)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Multi-task heads for each risk type (e.g., tumor progression, toxicity, other)
        self.output_heads = nn.ModuleList([nn.Linear(model_dim, 1) for _ in range(num_outputs)])

    def forward(self, x):
        """
        Args:
            x (Tensor): Input time-series tensor (B, T, input_dim)

        Returns:
            List[Tensor]: List of (B, 1) tensors for each risk type.
        """
        x = self.input_proj(x)           # (B, T, D)
        x = self.pos_encoder(x)          # add time encoding
        x = x.transpose(0, 1)            # (T, B, D)
        encoded = self.transformer(x)    # (T, B, D)
        final = encoded[-1]              # Use final timepoint

        return [head(final) for head in self.output_heads]
