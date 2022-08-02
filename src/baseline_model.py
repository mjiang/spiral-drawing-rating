import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import math
import sys
import numpy as np
from typing import Union, List, Dict, Any, cast
import argparse
import torch

from torchvision import transforms


epsilon = 1e-7

class baseline(nn.Module):
    def __init__(self):
        super(baseline, self).__init__()
        self.resnet = torchvision.models.resnet18(pretrained=True)
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])

        self.regression = nn.Linear(512, 1)
        self.flatten = nn.Flatten()

        self._initialize_weights()

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def forward(self, x):
        '''
        x [N, C, T]
        '''
        x = self.resnet(x)
        x = self.flatten(x)
        y = self.regression(x)

        return y

    def prediction(self, img):
        image = self.transform(img).unsqueeze(0)
        y = self.forward(image)

        return y

    def _initialize_weights(self):
        for m in self.regression.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)