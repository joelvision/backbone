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
    
class DenseBottleNeck(nn.Module):
    def __init__(self, in_c, growth_rate):
        super(DenseBottleNeck, self).__init__()
        inner_c= 4 * growth_rate
        self.residual= nn.Sequential(
            nn.BatchNorm2d(in_c),
            nn.ReLU(),
            nn.Conv2d(inner_c, inner_c, 1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inner_c),
            nn.ReLU(),
            nn.Conv2d(inner_c, growth_rate, 3, stride=1, padding=1, bias=False)       
        )
        
        self.short_cut= nn.Sequential()
        
    def forward(self, x):
        x= torch.cat([self.short_cut(x) + self.residual(x)], 1)
        return x
    
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()

        self.down_sample = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False),
            nn.AvgPool2d(2, stride=2)
        )

    def forward(self, x):
        return self.down_sample(x)