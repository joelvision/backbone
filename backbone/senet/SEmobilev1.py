import sys
sys.path.append("")
import torch
import torch.nn as nn

from layers.block import Depthwise
from layers.common import Conv

# MobileNetV1
class SEMobileNet(nn.Module):
    def __init__(self, width_multiplier, num_classes, is_seblock= True, init_weights=True):
        super().__init__()
        self.init_weights=init_weights
        alpha = width_multiplier

        self.conv1 = Conv(3, int(32*alpha), 3, 2, 1)
        self.conv2 = Depthwise(int(32*alpha), int(64*alpha), 1, is_seblock)
        # down sample
        self.conv3 = nn.Sequential(
            Depthwise(int(64*alpha), int(128*alpha), 2, is_seblock),
            Depthwise(int(128*alpha), int(128*alpha), 1, is_seblock)
        )
        # down sample
        self.conv4 = nn.Sequential(
            Depthwise(int(128*alpha), int(256*alpha), 2, is_seblock),
            Depthwise(int(256*alpha), int(256*alpha), 1, is_seblock)
        )
        # down sample
        self.conv5 = nn.Sequential(
            Depthwise(int(256*alpha), int(512*alpha), 2, is_seblock),
            Depthwise(int(512*alpha), int(512*alpha), 1, is_seblock),
            Depthwise(int(512*alpha), int(512*alpha), 1, is_seblock),
            Depthwise(int(512*alpha), int(512*alpha), 1, is_seblock),
            Depthwise(int(512*alpha), int(512*alpha), 1, is_seblock),
            Depthwise(int(512*alpha), int(512*alpha), 1, is_seblock),
        )
        # down sample
        self.conv6 = nn.Sequential(
            Depthwise(int(512*alpha), int(1024*alpha), 2, is_seblock)
        )
        # down sample
        self.conv7 = nn.Sequential(
            Depthwise(int(1024*alpha), int(1024*alpha), 2, is_seblock)
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.linear = nn.Linear(int(1024*alpha), num_classes)

        # weights initialization
        if self.init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

    # weights initialization function
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

def semobilenet(alpha=1, num_classes=10):
    return SEMobileNet(alpha, num_classes)



if __name__ == '__main__':
    model= semobilenet()
    x= torch.randn((3, 3, 224, 224))
    output= model(x)
    print(output.size())