# domain_adapted_fusion_model.py
# This model enables transfer learning by incorporating a cancer-type embedding vector
# to adapt treatment response predictions across different cancer datasets.

import torch
import torch.nn as nn
from .imaging_model import ImagingModel
from .genomics_model import GenomicsModel

class DomainAdaptedFusionModel(nn.Module):
    def __init__(self, image_dim=128, gene_dim=128, domain_dim=8, hidden_dim=64, num_classes=2):
        """
        Fusion model with cancer-type/domain embedding.

        Args:
            image_dim (int): Image feature vector size.
            gene_dim (int): Gene expression vector size.
            domain_dim (int): Number of cancer types (for embedding).
            hidden_dim (int): Hidden layer dimension.
            num_classes (int): Output classes (e.g. responder / non-responder).
        """
        super().__init__()

        self.image_encoder = ImagingModel(output_dim=image_dim)
        self.gene_encoder = GenomicsModel(output_dim=gene_dim)

        # Learnable cancer-type embedding
        self.domain_embedding = nn.Embedding(domain_dim, 16)

        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(image_dim + gene_dim + 16, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image, gene, domain_id):
        """
        Forward pass with domain-specific embedding.

        Args:
            image (Tensor): Image tensor (B, C, H, W)
            gene (Tensor): Gene expression vector (B, G)
            domain_id (Tensor): Cancer type indices (B,)

        Returns:
            Tensor: Output logits (B, num_classes)
        """
        image_feat = self.image_encoder(image)
        gene_feat = self.gene_encoder(gene)
        domain_feat = self.domain_embedding(domain_id)

        fused = torch.cat([image_feat, gene_feat, domain_feat], dim=1)
        return self.classifier(fused)
