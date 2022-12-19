import torch.nn as nn

class Swish(nn.Module):
    def __init__(self) -> None:
        super(Swish, self).__init__()
        self.sigmoid= nn.Sigmoid()
    
    def forward(self, x):
        return x * self.sigmoid(x)
    