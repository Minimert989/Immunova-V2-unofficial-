# immune_signatures.py
# This module selects and processes known immune escape or immune response gene signatures.
# It can be used to extract relevant features from RNA-seq data prior to model input.

import torch
import torch.nn as nn

# Example: Selected immune escape-related genes (placeholder - replace with validated gene sets)
IMMUNE_EVASION_GENES = [
    "CD274",  # PD-L1
    "PDCD1",  # PD-1
    "CTLA4",
    "LAG3",
    "TIGIT",
    "IDO1",
    "STAT1",
    "CXCL9",
    "CXCL10",
    "IFNG"
]

class ImmuneSignatureEncoder(nn.Module):
    def __init__(self, gene_indices, output_dim=32):
        """
        Extracts and compresses immune-related gene expression features.

        Args:
            gene_indices (List[int]): Indices of selected signature genes within input expression vector.
            output_dim (int): Output feature dimension after encoding.
        """
        super().__init__()
        self.gene_indices = gene_indices
        self.encoder = nn.Sequential(
            nn.Linear(len(gene_indices), 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )

    def forward(self, x):
        """
        Extracts signature genes and encodes them.

        Args:
            x (Tensor): Full gene expression vector (B, G)

        Returns:
            Tensor: Signature embedding vector (B, output_dim)
        """
        sig_input = x[:, self.gene_indices]
        return self.encoder(sig_input)
