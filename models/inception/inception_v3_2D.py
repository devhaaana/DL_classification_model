from typing import Any, Callable, List, Optional, Tuple

import warnings
from collections import namedtuple

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torchvision.utils import _log_api_usage_once
from torchvision.models.inception import BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux
from torchvision.models.inception import InceptionOutputs, Inception_V3_Weights
from torchvision.models._api import register_model, Weights, WeightsEnum
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface


class Inception3(nn.Module):
    def __init__(
        self,
        input_dim,
        target_dim,
        aux_logits: bool = True,
        transform_input: bool = False,
        inception_blocks: Optional[List[Callable[..., nn.Module]]] = None,
        init_weights: Optional[bool] = None,
        dropout: float = 0.5,
    ) -> None:
        super(Inception3, self).__init__()
        _log_api_usage_once(self)
        
        if inception_blocks is None:
            inception_blocks = [BasicConv2d, InceptionA, InceptionB, InceptionC, InceptionD, InceptionE, InceptionAux]
        if init_weights is None:
            warnings.warn(
                "The default weight initialization of inception_v3 will be changed in future releases of "
                "torchvision. If you wish to keep the old behavior (which leads to long initialization times"
                " due to scipy/scipy#11299), please set init_weights=True.",
                FutureWarning,
            )
            init_weights = True
        if len(inception_blocks) != 7:
            raise ValueError(f"length of inception_blocks should be 7 instead of {len(inception_blocks)}")
        
        conv_block = inception_blocks[0]
        inception_a = inception_blocks[1]
        inception_b = inception_blocks[2]
        inception_c = inception_blocks[3]
        inception_d = inception_blocks[4]
        inception_e = inception_blocks[5]
        inception_aux = inception_blocks[6]
        
        in_channels = 3 # RGB
        num_classes = target_dim

        self.aux_logits = aux_logits
        self.transform_input = transform_input
        
        self.Conv2d_1a_3x3 = conv_block(in_channels=in_channels, out_channels=32, kernel_size=3, stride=2)
        self.Conv2d_2a_3x3 = conv_block(in_channels=32, out_channels=32, kernel_size=3)
        self.Conv2d_2b_3x3 = conv_block(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.Conv2d_3b_1x1 = conv_block(in_channels=64, out_channels=80, kernel_size=1)
        self.Conv2d_4a_3x3 = conv_block(in_channels=80, out_channels=192, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=3, stride=2)
        
        self.Mixed_5b = inception_a(in_channels=192, pool_features=32)
        self.Mixed_5c = inception_a(in_channels=256, pool_features=64)
        self.Mixed_5d = inception_a(in_channels=288, pool_features=64)
        
        self.Mixed_6a = inception_b(in_channels=288)
        self.Mixed_6b = inception_c(in_channels=768, channels_7x7=128)
        self.Mixed_6c = inception_c(in_channels=768, channels_7x7=160)
        self.Mixed_6d = inception_c(in_channels=768, channels_7x7=160)
        self.Mixed_6e = inception_c(in_channels=768, channels_7x7=192)
        
        self.AuxLogits: Optional[nn.Module] = None
        if aux_logits:
            self.AuxLogits = inception_aux(in_channels=768, num_classes=num_classes)
            
        self.Mixed_7a = inception_d(in_channels=768)
        self.Mixed_7b = inception_e(in_channels=1280)
        self.Mixed_7c = inception_e(in_channels=2048)
        
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(in_features=2048, out_features=num_classes)
        
        if init_weights:
            for m in self.modules():
                if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                    stddev = float(m.stddev) if hasattr(m, "stddev") else 0.1  # type: ignore
                    torch.nn.init.trunc_normal_(m.weight, mean=0.0, std=stddev, a=-2, b=2)
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def _transform_input(self, x: Tensor) -> Tensor:
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[:, 0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[:, 1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[:, 2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x: Tensor) -> Tuple[Tensor, Optional[Tensor]]:
        x = self.Conv2d_1a_3x3(x)
        x = self.Conv2d_2a_3x3(x)
        x = self.Conv2d_2b_3x3(x)
        x = self.maxpool1(x)
        
        x = self.Conv2d_3b_1x1(x)
        x = self.Conv2d_4a_3x3(x)
        x = self.maxpool2(x)
        
        x = self.Mixed_5b(x)
        x = self.Mixed_5c(x)
        x = self.Mixed_5d(x)
        
        x = self.Mixed_6a(x)
        x = self.Mixed_6b(x)
        x = self.Mixed_6c(x)
        x = self.Mixed_6d(x)
        x = self.Mixed_6e(x)

        aux: Optional[Tensor] = None
        if self.AuxLogits is not None:
            if self.training:
                aux = self.AuxLogits(x)

        x = self.Mixed_7a(x)
        x = self.Mixed_7b(x)
        x = self.Mixed_7c(x)
        
        x = self.avgpool(x)
        x = self.dropout(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x, aux

    @torch.jit.unused
    def eager_outputs(self, x: Tensor, aux: Optional[Tensor]) -> InceptionOutputs:
        if self.training and self.aux_logits:
            return InceptionOutputs(x, aux)
        else:
            return x  # type: ignore[return-value]

    def forward(self, x: Tensor) -> InceptionOutputs:
        x = self._transform_input(x)
        x, aux = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted Inception3 always returns Inception3 Tuple")
            return InceptionOutputs(x, aux)
        else:
            return self.eager_outputs(x, aux)



@register_model()
@handle_legacy_interface(weights=("pretrained", Inception_V3_Weights.IMAGENET1K_V1))
def Model(*, weights: Optional[Inception_V3_Weights] = None, progress: bool = True, **kwargs: Any) -> Inception3:

    weights = Inception_V3_Weights.verify(weights)

    original_aux_logits = kwargs.get("aux_logits", True)
    if weights is not None:
        if "transform_input" not in kwargs:
            _ovewrite_named_param(kwargs, "transform_input", True)
        _ovewrite_named_param(kwargs, "aux_logits", True)
        _ovewrite_named_param(kwargs, "init_weights", False)
        _ovewrite_named_param(kwargs, "target_dim", len(weights.meta["categories"]))

    model = Inception3(**kwargs)

    if weights is not None:
        model.load_state_dict(weights.get_state_dict(progress=progress, check_hash=True))
        if not original_aux_logits:
            model.aux_logits = False
            model.AuxLogits = None

    return model

