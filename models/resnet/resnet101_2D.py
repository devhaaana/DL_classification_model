import os, sys
from torchvision.models.resnet import *

def Model(input_dim, target_dim, **kwargs):
    num_classes = target_dim
    return resnet101(num_classes=num_classes, **kwargs)