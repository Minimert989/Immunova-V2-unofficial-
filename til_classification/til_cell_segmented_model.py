# til_cell_segmented_model.py
# This module defines a classifier that integrates cell-level segmentation features (e.g., from HoVerNet)
# with image patches to improve TIL subtype classification.

import torch
import torch.nn as nn
import torch.nn.functional as F

class TILCellIntegratedClassifier(nn.Module):
    def __init__(self, patch_feature_dim=256, cell_feature_dim=64, hidden_dim=128, num_classes=4):
        """
        Initializes a hybrid classifier that combines patch features with cell segmentation features.

        Args:
            patch_feature_dim (int): Feature dimension extracted from image patch (e.g., ResNet).
            cell_feature_dim (int): Aggregated feature from segmented cells (e.g., from HoVerNet).
            hidden_dim (int): Hidden layer size.
            num_classes (int): Number of output TIL subtype classes.
        """
        super().__init__()

        # Fusion layer to combine patch + cell features
        self.fusion_layer = nn.Sequential(
            nn.Linear(patch_feature_dim + cell_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, patch_feat, cell_feat):
        """
        Forward pass.

        Args:
            patch_feat (Tensor): Image patch embedding (B, patch_feature_dim)
            cell_feat (Tensor): Cell-level aggregated embedding (B, cell_feature_dim)

        Returns:
            Tensor: Logits for TIL subtype prediction (B, num_classes)
        """
        combined = torch.cat([patch_feat, cell_feat], dim=1)
        return self.fusion_layer(combined)
