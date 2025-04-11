# model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# Basic CNN Model for TIL Classification
class TILClassifier(nn.Module):
    def __init__(self, in_channels=3, num_classes=2):
        super(TILClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.25)
        self.fc1 = nn.Linear(64 * 32 * 32, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # [B, 32, H/2, W/2]
        x = self.pool(F.relu(self.conv2(x)))   # [B, 64, H/4, W/4]
        x = x.view(x.size(0), -1)              # flatten
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Transformer model for survival prediction
class TransformerSurvivalModel(nn.Module):
    def __init__(self, input_dim, num_heads, num_layers, dropout=0.1):
        super(TransformerSurvivalModel, self).__init__()
        self.input_linear = nn.Linear(input_dim, 128)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=num_heads, dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output = nn.Linear(128, 1)

    def forward(self, x):
        x = self.input_linear(x)
        x = self.transformer(x)
        x = self.output(x.mean(dim=0))  # mean over sequence length
        return x


# Genomics + Imaging Fusion Model
class FusionModel(nn.Module):
    def __init__(self, genomics_dim, img_dim):
        super(FusionModel, self).__init__()
        self.genomics_fc = nn.Sequential(
            nn.Linear(genomics_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.img_fc = nn.Sequential(
            nn.Linear(img_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, genomics_data, img_data):
        x1 = self.genomics_fc(genomics_data)
        x2 = self.img_fc(img_data)
        x = torch.cat((x1, x2), dim=1)
        out = self.classifier(x)
        return out
