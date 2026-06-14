"""
Small residual CNN for CIFAR-10 experiments.
"""
from torch import nn


def _activation(name):
    name = name.lower()
    if name == 'relu':
        return nn.ReLU(inplace=True)
    if name == 'leaky_relu':
        return nn.LeakyReLU(negative_slope=0.1, inplace=True)
    if name == 'elu':
        return nn.ELU(inplace=True)
    if name == 'tanh':
        return nn.Tanh()
    raise ValueError(f'Unknown activation: {name}')


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation='relu'):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            _activation(activation),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        if in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        self.out_activation = _activation(activation)

    def forward(self, x):
        return self.out_activation(self.conv(x) + self.shortcut(x))


class CIFARResCNN(nn.Module):
    """Compact custom CNN for the CIFAR-10 scoring task."""

    def __init__(self, num_classes=10, widths=(32, 64, 128), blocks_per_stage=2,
                 activation='relu', dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            _activation(activation),
        )
        self.stage1 = self._make_stage(widths[0], widths[0], blocks_per_stage, activation)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage2 = self._make_stage(widths[0], widths[1], blocks_per_stage, activation)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage3 = self._make_stage(widths[1], widths[2], blocks_per_stage, activation)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[2], num_classes),
        )

    def _make_stage(self, in_channels, out_channels, blocks_per_stage, activation):
        layers = [ResidualBlock(in_channels, out_channels, activation)]
        for _ in range(blocks_per_stage - 1):
            layers.append(ResidualBlock(out_channels, out_channels, activation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.pool1(x)
        x = self.stage2(x)
        x = self.pool2(x)
        x = self.stage3(x)
        return self.head(x)
