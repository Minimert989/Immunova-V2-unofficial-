# treatment_response/genomics_model.py

import torch
import torch.nn as nn

class GenomicsModel(nn.Module):
    def __init__(self, input_dim=1000, hidden_dim=256, output_dim=128):
        super(GenomicsModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.model(x)
