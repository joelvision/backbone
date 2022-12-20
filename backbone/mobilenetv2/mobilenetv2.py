import numpy as np
import torch
import torch.nn as nn

from layers.common import Conv
from layers.block import InvertedResidual

class MobileNetV2(nn.Module):
    def __init__(self, in_c, nc):
        super(MobileNetV2, self).__init__()
        self.configs=[
            # t, c, n, s
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]
        self.stem= Conv(in_c, 32, 3, 2, None)
        layers = []
        input_channel = 32
        for t, c, n, s in self.configs:
            for i in range(n):
                s = s if i == 0 else 1
                layers.append(InvertedResidual(in_c= input_channel, out_c= c, expand_ratio= t))
                input_channel = c

        self.layers = nn.Sequential(*layers)
        
        self.last_conv= Conv(input_channel, 1280, 3, 1, 0, act=False)      
        self.classifier = nn.Sequential(
            nn.Dropout2d(0.2),
            nn.Linear(1280, nc)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.stem_conv(x)
        x = self.layers(x)
        x = self.last_conv(x)
        x = self.avg_pool(x).view(-1, 1280)
        x = self.classifier(x)
        return x