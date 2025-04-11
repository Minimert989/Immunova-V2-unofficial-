# Grad-CAM implementation
# Uses hook to extract gradients
import torch

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.hook()

    def hook(self):
        def forward_hook(module, input, output):
            self.gradients = output
        self.model.layer4[1].register_forward_hook(forward_hook)