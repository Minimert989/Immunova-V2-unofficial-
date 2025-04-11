# fusion_attention_model.py
# This model performs cross-modal attention fusion between image and gene expression features.
# The model learns to weight features from each modality when predicting treatment response.

import torch
import torch.nn as nn
from .imaging_model import ImagingModel
from .genomics_model import GenomicsModel

class CrossModalFusion(nn.Module):
    def __init__(self, image_dim=128, gene_dim=128, hidden_dim=64, num_classes=2):
        """
        Initializes the cross-modal attention fusion model.

        Args:
            image_dim (int): Image feature vector dimension.
            gene_dim (int): Genomic feature vector dimension.
            hidden_dim (int): Hidden layer size.
            num_classes (int): Number of output classes (e.g., response vs no response).
        """
        super().__init__()

        self.image_encoder = ImagingModel(output_dim=image_dim)
        self.genomic_encoder = GenomicsModel(output_dim=gene_dim)

        # Learnable attention weights per modality
        self.image_weight = nn.Linear(image_dim, 1)
        self.gene_weight = nn.Linear(gene_dim, 1)

        # Classifier after attention-weighted fusion
        self.classifier = nn.Sequential(
            nn.Linear(image_dim + gene_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image, genomic):
        """
        Forward pass with cross-modal attention fusion.

        Args:
            image (Tensor): Image input tensor (B, C, H, W)
            genomic (Tensor): Gene expression vector (B, G)

        Returns:
            Tensor: Output logits (B, num_classes)
        """
        image_feat = self.image_encoder(image)     # (B, image_dim)
        gene_feat = self.genomic_encoder(genomic)  # (B, gene_dim)

        # Compute modality attention scores
        alpha_img = torch.sigmoid(self.image_weight(image_feat))  # (B, 1)
        alpha_gene = torch.sigmoid(self.gene_weight(gene_feat))   # (B, 1)

        # Normalize attention
        alpha_sum = alpha_img + alpha_gene + 1e-6
        alpha_img = alpha_img / alpha_sum
        alpha_gene = alpha_gene / alpha_sum

        # Weighted fusion
        fused = torch.cat([image_feat * alpha_img, gene_feat * alpha_gene], dim=1)
        return self.classifier(fused)
