import torch
from torch import nn
from torchinfo import summary


class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        inner_channels = 4 * growth_rate

        self.residual = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, inner_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inner_channels),
            nn.ReLU(),
            nn.Conv2d(inner_channels, growth_rate, kernel_size=3, stride=1, padding=1, bias=False),
        )
        
        self.shortcut = nn.Sequential()
        
    def forward(self, x):
        return torch.cat([self.shortcut(x), self.residual(x)], dim = 1)
    
class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.transition = nn.Sequential(
            nn.BatchNorm2d(in_channels), # Dense block end: Conv
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias = False), # Reduce channels
            # nn.Conv2d(in_channels, int(in_channels / 2), 1, bias = False), # Reduce channels
            nn.AvgPool2d(2, stride=2),    # Reduce feature map size
        )
    
    def forward(self, x):
        return self.transition(x)