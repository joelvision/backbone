import torch
import torch.nn as nn

from layers.activation import Swish

class SEBlock(nn.Module):
    def __init__(self, in_c, r=4):
        super(SEBlock, self).__init__()
        self.squeeze= nn.AdaptiveAvgPool2d((1,1))
        self.excition= nn.Sequential(
            nn.Linear(in_c, in_c * r),
            Swish(),
            nn.Linear(in_c * r, in_c),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x= self.squeeze(x)
        x= x.view(x.size(0), -1)
        x= self.excition(x)
        x= x.view(x.size(0), x.size(1), 1, 1)
        
        return x
    