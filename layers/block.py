import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.common import Conv, ConvBnAct

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
    

class ResNextBottleNeck(nn.Module):
    mul= 2
    
    def __init__(self, in_c, group_width, cardinality, s= 1):
        super(ResNextBottleNeck, self).__init__()
        self.conv1= Conv(in_c, group_width, 1, s)
        self.conv2= Conv(group_width, group_width, 3, 1, None, cardinality)
        self.conv3= Conv(group_width, group_width * self.mul, 1, 1, act=False)

        self.shortcut= nn.Sequential()
        
        if s != 1 or in_c != group_width * self.mul:
            self.shortcut= nn.Sequential(
                Conv(in_c, group_width * self.mul, 1, s, act=False)
            )
    
    def forward(self, x):
        x= self.conv1(x)
        x= self.conv2(x)
        x= self.conv3(x)
        x+= self.shortcut(x)
        x= F.relu(x)
        
        return x

class Depthwise(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=None, g=1, act=True):
        super(Depthwise, self).__init__()
        self.depth= nn.Sequential(
            Conv(in_c, in_c, k, s, p, g, act)
        )
        
    def forward(self, x):
        return self.depth(x)

class Pointwise(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=None, g=1, act=True):
        self.pointwise= nn.Sequential(
            Conv(in_c, out_c, k, s, p, g, act)
        )
    
    def forward(self, x):
        return self.pointwise(x)
    
class Depthwise(nn.Module):
    def __init__(self, in_c, out_c, k=1, s=1, p=None, g=1, act=True, is_seblock=True):
        super(Depthwise, self).__init__()
        self.depthwise= nn.Sequential(
            Conv(in_c, in_c, 3, s, None, in_c)
        )
        
        self.pointwise= nn.Sequential(
            Conv(in_c, out_c, 1, 1, 0)
        )
        
        if is_seblock:
            self.seblock= SEblock(out_c)
        else:
            self.seblock= nn.Identity()
            
    def forward(self, x):
        x= self.depthwise(x)
        x= self.pointwise(x)
        x= self.seblock(x)
        return x
    
class SEblock(nn.Module):
    def __init__(self, in_c, r= 16):
        super(SEblock, self).__init__()
        self.squeeze= nn.AdaptiveAvgPool2d((1, 1))
        self.excitation= nn.Sequential(
            nn.Linear(in_c, in_c // r),
            nn.ReLU(),
            nn.Linear(in_c //r, in_c),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x= self.squeeze(x)
        x= x.view(x.size(0), -1)
        x= self.excitation(x)
        x= x.view(x.size(0), x.size(1), 1, 1)

        return x
    
class MbConv(nn.Module):
    def __init__(self, in_c, out_c, k=1, se_scale=4, p=0.5):
        super(MbConv, self)._init__()
        self.p= torch.tensor(p) if (in_c == out_c) else torch.tensor(1).float()