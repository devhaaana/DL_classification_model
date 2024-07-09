import torch
from torch import nn
from .googlenet_2D_function import *


class Model(nn.Module):
    def __init__(self, input_dim, target_dim, **kwargs):
        super(Model, self).__init__()
        
        self.input_dim = input_dim
        num_classes = target_dim
        
        if 'init_weights' in kwargs:
            self.init_weights = kwargs['init_weights']
        else:
            self.init_weights = True
            
        if 'aux_logits' in kwargs:
            self.aux_logits = kwargs['aux_logits']
        else:
            self.aux_logits = True
        
        # conv_block takes in_channels, out_channels, kernel_size, stride, padding
        # Inception block takes out1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool

        self.conv1 = conv_block(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = conv_block(in_channels=64, out_channels=192, kernel_size=3, stride=1, padding=1)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception3a = Inception_Module(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception_Module(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = Inception_Module(480, 192, 96, 208, 16, 48, 64)

        # auxiliary classifier
        self.inception4b = Inception_Module(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception_Module(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception_Module(512, 112, 144, 288, 32, 64, 64)

        # auxiliary classifier
        self.inception4e = Inception_Module(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.inception5a = Inception_Module(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception_Module(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AvgPool2d(1, 1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc1 = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = Auxiliary_Classifier(512, num_classes)
            self.aux2 = Auxiliary_Classifier(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

        if self.init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)
        
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        
        x = self.inception4a(x)
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)

        if self.aux_logits and self.training:
            aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.dropout(x)
        x = self.fc1(x)

        if self.aux_logits and self.training:
            return [x, aux1, aux2]
        else:
            return x 

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
