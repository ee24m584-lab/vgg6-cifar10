"""VGG6 for CIFAR-10 with a configurable activation function."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# name -> factory producing a fresh activation module
ACTIVATIONS = {
    "relu": lambda: nn.ReLU(inplace=True),
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
}

# conv widths; "M" = 2x2 max-pool
CFG_VGG6 = [64, 64, "M", 128, 128, "M"]


class VGG(nn.Module):
    def __init__(self, features: nn.Module, num_classes: int = 10):
        super().__init__()
        self.features = features
        self.classifier = nn.Linear(128, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def _make_layers(cfg, activation: str, batch_norm: bool) -> nn.Sequential:
    layers, in_ch = [], 3
    for v in cfg:
        if v == "M":
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            continue
        layers.append(nn.Conv2d(in_ch, v, kernel_size=3, padding=1))
        if batch_norm:
            layers.append(nn.BatchNorm2d(v))
        layers.append(ACTIVATIONS[activation]())
        in_ch = v
    return nn.Sequential(*layers)


def build_vgg6(activation: str = "relu", batch_norm: bool = True,
               num_classes: int = 10) -> VGG:
    if activation not in ACTIVATIONS:
        raise ValueError(f"unknown activation '{activation}', choose from {list(ACTIVATIONS)}")
    return VGG(_make_layers(CFG_VGG6, activation, batch_norm), num_classes)
