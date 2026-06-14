"""
Plain CNN baseline for CIFAR-10 experiments.
"""
from torch import nn

from .rescnn import _activation


def _conv_block(in_channels, out_channels, activation):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        _activation(activation),
    )


class PlainCNN(nn.Module):
    """Non-residual CNN baseline with the same width/depth knobs as ResCNN."""

    def __init__(self, num_classes=10, widths=(32, 64, 128), blocks_per_stage=2,
                 activation='relu', dropout=0.2):
        super().__init__()
        stages = []
        in_channels = 3
        for stage_idx, width in enumerate(widths):
            blocks = [_conv_block(in_channels, width, activation)]
            for _ in range(blocks_per_stage - 1):
                blocks.append(_conv_block(width, width, activation))
            stages.append(nn.Sequential(*blocks))
            if stage_idx < len(widths) - 1:
                stages.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = width

        self.features = nn.Sequential(*stages)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(widths[-1], num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        return self.head(x)
