import sys
sys.path.append("")
import torch
import torch.nn as nn
from classification.layers.activation import Swish

def autopad(k, p=None):
    # pad to same
    
    if p is None:
        p= k //2 if isinstance(k, int) else [x//2 for x in k]
    return p

class MP(nn.Module):
    def __init__(self, k, s, **args):
        super(MP, self).__init__()
        self.m= nn.MaxPool2d(kernel_size= k, stride= k, **args)
        
    def forward(self, x):
        return self.m(x)
    
    
class Conv(nn.Module):
    def __init__(self, in_c, out_c, k, s, p=None, g=1 , act= True):
        super().__init__()
        self.conv= nn.Conv2d(in_c, out_c, k, s, autopad(k, p), groups= g, bias= False)
        self.bn= nn.BatchNorm2d(out_c)
        self.act= nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())
        
    def forward(self, x):
        x= self.conv(x)
        x= self.bn(x)
        x= self.act(x)
        
        return x
    

