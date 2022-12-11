import sys
sys.path.append('')
import torch
import torch.nn as nn
from layers.common import Conv, MP

class VGG(nn.Module):
    def __init__(self, in_c, nc, info, init_weight=True):
        super(VGG, self).__init__()
        self.in_c= in_c
        self.vgg_layer= self.create_conv_layer(info)
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, nc),
        )

        if init_weight:
            self._initialize_weights()
    
    def forward(self,x):
        x= self.vgg_layer(x)
        x= x.view(-1, 512 * 7 * 7)
        x= self.fc(x)
        
        return x
    
    def create_conv_layer(self, info):
        layers= list()
        in_c= self.in_c
        
        for x in info:
            if type(x) == int:
                out_c= x
                layers+= [Conv(in_c= in_c, out_c= out_c, k= 3, s= 1, p= None, g= 1, act= True)]
                in_c= x   
            elif x == 'M':
                layers+= [MP(k=2)]
        
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
                
def VGG11(nc):
    vgg11= [64, 'M', 128, 'M', 256, 256, 'M', 512,512, 'M',512,512,'M']
    model= VGG(in_c=3, nc= nc, info=vgg11)
    return model

def VGG13(nc):
    vgg13= [64,64, 'M', 128, 128, 'M', 256, 256, 'M', 512,512, 'M', 512,512,'M']
    model= VGG(in_c=3, nc= nc, info=vgg13)
    return model

def VGG16(nc):
    vgg16= [64,64, 'M', 128, 128, 'M', 256, 256,256, 'M', 512,512,512, 'M',512,512,512,'M']
    model= VGG(in_c=3, nc= nc, info=vgg16)
    return model

def VGG19(nc):
    vgg19= [64,64, 'M', 128, 128, 'M', 256, 256,256,256, 'M', 512,512,512,512, 'M',512,512,512,512,'M']
    model= VGG(in_c=3, nc= nc, info=vgg19)
    return model