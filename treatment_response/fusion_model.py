# treatment_response/fusion_model.py

import torch
import torch.nn as nn
from .imaging_model import ImagingModel
from .genomics_model import GenomicsModel

class FusionModel(nn.Module):
    def __init__(self, image_dim=128, genomic_dim=128, hidden_dim=64, num_classes=2):
        super(FusionModel, self).__init__()
        self.image_encoder = ImagingModel(output_dim=image_dim)
        self.genomic_encoder = GenomicsModel(output_dim=genomic_dim)
        self.classifier = nn.Sequential(
            nn.Linear(image_dim + genomic_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, image, genomic):
        image_feat = self.image_encoder(image)
        genomic_feat = self.genomic_encoder(genomic)
        combined = torch.cat((image_feat, genomic_feat), dim=1)
        return self.classifier(combined)
