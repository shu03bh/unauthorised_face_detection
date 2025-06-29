# model.py

import torch
import torch.nn as nn
from torchvision import models

class HybridAlexResNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.alex = models.alexnet(weights=models.AlexNet_Weights.DEFAULT).features
        self.res = nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-1])
        self.classifier = nn.Sequential(
            nn.Linear(256 * 6 * 6 + 512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)  # no sigmoid
        )

    def forward(self, x):
        a = torch.flatten(self.alex(x), 1)
        r = torch.flatten(self.res(x), 1)
        x = torch.cat((a, r), dim=1)
        return self.classifier(x)

def load_model(path='hybrid_model.pth'):
    model = HybridAlexResNet()
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    return model
