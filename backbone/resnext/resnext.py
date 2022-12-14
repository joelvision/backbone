import sys
sys.path.append('')

import torch
import torch.nn as nn

from layers.common import Conv, MP
from layers.block import ResNextBottleNeck


class ResNext(nn.Module):
    def __init__(self, block, num_blocks, nc, c= 32, w= 4):
        super(ResNext, self).__init__()
        self.in_c= 64
        self.group_conv= c * w
        
        self.stem= nn.Sequential(
            Conv(3, self.in_c, 7, 2, 3),
            MP(k=3, s=2, padding=1)
        )
        
        self.layer1= self._create_layer(block, c, num_blocks[0], s= 1)
        self.layer2= self._create_layer(block, c, num_blocks[1], s= 2)
        self.layer3= self._create_layer(block, c, num_blocks[2], s= 2)
        self.layer4= self._create_layer(block, c, num_blocks[3], s= 2)
        self.avgpool= nn.AdaptiveAvgPool2d((1,1))
        self.linear= nn.Linear(self.group_conv, nc)
    
    def forward(self, x):
        x= self.stem(x)
        x= self.layer1(x)
        x= self.layer2(x)
        x= self.layer3(x)
        x= self.layer4(x)
        x= self.avgpool(x)
        x= torch.flatten(x, 1)
        x= self.linear(x)
        return x
        
    def _create_layer(self, block, c, num_blocks, s):
        strides= [s] + [1] * (num_blocks - 1)
        layers= []
        
        for i in range(num_blocks):
            layers.append(block(self.in_c, self.group_conv, c, strides[i]))
        self.in_c= block.mul * self.group_conv
        
        return nn.Sequential(*layers)    
            
def ResNeXt50():
    return ResNext(ResNextBottleNeck, [3, 4, 6, 3], nc= 10)

    
def ResNeXt101():
    return ResNext(ResNextBottleNeck, [3, 4, 23, 3], nc= 10)

def ResNeXt152():
    return ResNext(ResNextBottleNeck, [3, 8, 36, 3], nc= 10)
        
if __name__ == '__main__':
    x= torch.randn((1, 3, 224, 224))
    model= ResNeXt50()
    output= model(x)
    print(output.size())  