import torch
import torch.nn as nn

from layers.common import Conv

class ResnetBlock(nn.Module):
    expansion=1
    def __init__(self, in_c, out_c, s= 1):
        super(ResnetBlock, self).__init__()
        self.residual_block= nn.Sequential(
            Conv(in_c, out_c, k=3, s=s, p=None),
            Conv(out_c, out_c * ResnetBlock.expansion, k= 3, s= 1, p= None, act= False)
        )
        self.short_cut= nn.Sequential()
        self.act= nn.ReLU()
        
        if s != 1 or in_c != ResnetBlock.expansion * out_c :
            self.short_cut= nn.Sequential(
                Conv(in_c, out_c, k=1, s=s, p=None, act=False)
            )
            
    def forward(self, x):
        x= self.residual_block(x) + self.short_cut(x)
        x= self.act(x)
        
        return x
    
    
class ResnetBottleNeckBlock(nn.Module):
    expansion= 4
    def __init__(self, in_c, out_c, s= 1):
        super(ResnetBottleNeckBlock, self).__init__()
        self.residual_block= nn.Sequential(
            Conv(in_c, out_c, k= 1, s=1),
            Conv(out_c, out_c, k= 3, s= s),
            Conv(out_c, out_c * ResnetBottleNeckBlock.expansion, k= 1, s= 1, act=False)
        )
        self.short_cut= nn.Sequential()
        self.act= nn.ReLU()
        
        if s != 1 or in_c != out_c * ResnetBottleNeckBlock.expansion:
            self.short_cut(
                Conv(in_c, out_c * ResnetBottleNeckBlock.expansion, k= 1, s=s, act=False)
            )
    
    def forward(self, x):
        x= self.residual_block(x) + self.short_cut(x)
        x= self.act(x)
        
        return x
    
        