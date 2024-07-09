import torch
from torch import nn


class conv_block(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(conv_block, self).__init__()

        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, **kwargs),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )
    
    def forward(self, x):
        return self.conv_layer(x)


class Inception_Module(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool):
        super(Inception_Module, self).__init__()

        self.branch1 = conv_block(in_channels=in_channels, out_channels=out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=red_3x3, kernel_size=1),
            conv_block(in_channels=red_3x3, out_channels=out_3x3, kernel_size=3, padding=1),
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels=in_channels, out_channels=red_5x5, kernel_size=1),
            conv_block(in_channels=red_5x5, out_channels=out_5x5, kernel_size=5, padding=2),
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels=in_channels, out_channels=out_1x1pool, kernel_size=1)
        )

    def forward(self, x):
        x = torch.cat([self.branch1(x), self.branch2(x), self.branch3(x), self.branch4(x)], 1)
        return x

class Auxiliary_Classifier(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Auxiliary_Classifier, self).__init__()

        self.conv = nn.Sequential(
            nn.AvgPool2d(kernel_size=5, stride=3),
            conv_block(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1, padding=0),
        )

        self.fc = nn.Sequential(
            nn.Linear(in_features=2048, out_features=1024),
            nn.ReLU(),
            nn.Dropout(p=0.7),
            nn.Linear(in_features=1024, out_features=num_classes),
        )

    def forward(self,x):
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x