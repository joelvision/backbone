import sys
sys.path.append('')
import torch
import torch.nn as nn

from layers.common import Conv, MP
from layers.block import ResnetBlock, ResnetBottleNeckBlock

class ResNet(nn.Module):
    def __init__(self, block, num_block, nc= 100, init_weights= True):
        super(ResNet, self).__init__()
        self.in_c= 64
        self.stem= nn.Sequential(
            Conv(3, 64, 7, 2, 3),
            nn.MaxPool2d(3, 2, 1)
        )
        
        self.block1= self.create_layers(block, 64, num_block[0], 1)
        self.block2= self.create_layers(block, 128, num_block[1], 2)
        self.block3= self.create_layers(block, 256, num_block[2], 2)
        self.block4= self.create_layers(block, 512, num_block[3], 2)
        
        self.avg_pool= nn.AdaptiveAvgPool2d((1,1))
        self.fc= nn.Linear(512 * block.expansion, nc)
        

        if init_weights:
            self._initialize_weights()
            
    def create_layers(self, block, out_c, num_blocks, s):
        strides= [s] + [1] * (num_blocks - 1)
        layers= []
        
        for stride in strides:
            layers.append(block(self.in_c, out_c, stride))
            self.in_c= out_c * block.expansion
            
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x= self.stem(x)
        x= self.block1(x)
        x= self.block2(x)
        x= self.block3(x)
        x= self.block4(x)
        x= self.avg_pool(x)
        x= x.view(x.size(0), -1)
        x= self.fc(x)
        
        return x

def resnet18():
    return ResNet(ResnetBlock, [2,2,2,2])

def resnet34():
    return ResNet(ResnetBlock, [3, 4, 6, 3])

def resnet50():
    return ResNet(ResnetBottleNeckBlock, [3,4,6,3])

def resnet101():
    return ResNet(ResnetBottleNeckBlock, [3, 4, 23, 3])

def resnet152():
    return ResNet(ResnetBottleNeckBlock, [3, 8, 36, 3])


if __name__ == '__main__':
    x= torch.randn((1, 3, 224, 224))
    model= resnet18()
    output= model(x)
    print(output.size())