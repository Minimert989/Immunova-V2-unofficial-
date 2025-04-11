# treatment_response/imaging_model.py

import torch
import torch.nn as nn
import torchvision.models as models

class ImagingModel(nn.Module):
    def __init__(self, output_dim=128):
        super(ImagingModel, self).__init__()
        base = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, output_dim)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)
