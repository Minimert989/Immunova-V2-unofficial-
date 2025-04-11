# til_multiclass_model.py
# Multiclass classification of TIL subtypes: CD8+, CD4+, Treg, None
# Built on a ResNet backbone with Dropout, BatchNorm, and softmax output

import torch
import torch.nn as nn
import torchvision.models as models

TIL_CLASSES = ["None", "CD8+", "CD4+", "Treg"]

class TILMulticlassClassifier(nn.Module):
    def __init__(self, num_classes=len(TIL_CLASSES)):
        """
        Multiclass TIL classifier using ResNet18 backbone.

        Args:
            num_classes (int): Number of TIL subtypes to classify.
        """
        super().__init__()

        # Load pretrained ResNet18 as backbone
        base_model = models.resnet18(pretrained=True)

        # Freeze early layers (optional)
        for param in base_model.parameters():
            param.requires_grad = True  # Change to False if you want to freeze early layers

        # Replace final classification head
        in_features = base_model.fc.in_features
        base_model.fc = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Dropout(0.5),
            nn.Linear(in_features, num_classes)
        )

        self.model = base_model

    def forward(self, x):
        """
        Args:
            x (Tensor): Input image tensor (B, C, H, W)

        Returns:
            Tensor: Softmax scores (B, num_classes)
        """
        return self.model(x)
