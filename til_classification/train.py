# Training pipeline for TIL classifier
import torch
from torchvision import datasets, transforms

def train():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FakeData(transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4)
    for imgs, labels in loader:
        print('Training batch')