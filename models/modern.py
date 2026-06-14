"""
Hybrid non-CNN model families for CIFAR-10 comparison.
"""
import torch
from torch import nn

from .rescnn import _activation


def _num_heads(embed_dim):
    for heads in (8, 6, 4, 3, 2):
        if embed_dim % heads == 0:
            return heads
    return 1


def _conv_stem(embed_dim, activation):
    hidden = max(16, embed_dim // 2)
    return nn.Sequential(
        nn.Conv2d(3, hidden, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(hidden),
        _activation(activation),
        nn.MaxPool2d(kernel_size=2, stride=2),
        nn.Conv2d(hidden, embed_dim, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(embed_dim),
        _activation(activation),
        nn.MaxPool2d(kernel_size=2, stride=2),
    )


class ConvTokenTransformer(nn.Module):
    """Small transformer classifier with a convolutional tokenizer."""

    def __init__(self, num_classes=10, widths=(48, 96, 192), blocks_per_stage=2,
                 activation='relu', dropout=0.2):
        super().__init__()
        embed_dim = widths[-1]
        num_tokens = 8 * 8
        self.tokenizer = _conv_stem(embed_dim, activation)
        self.position = nn.Parameter(torch.zeros(1, num_tokens, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=_num_heads(embed_dim),
            dim_feedforward=max(widths[-1] * 2, widths[1] * 2),
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, blocks_per_stage))
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        x = self.tokenizer(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.encoder(x + self.position)
        return self.head(x.mean(dim=1))


class MixerBlock(nn.Module):
    def __init__(self, num_tokens, embed_dim, token_dim, channel_dim, activation='relu', dropout=0.2):
        super().__init__()
        self.token_norm = nn.LayerNorm(embed_dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, token_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(token_dim, num_tokens),
            nn.Dropout(dropout),
        )
        self.channel_norm = nn.LayerNorm(embed_dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(embed_dim, channel_dim),
            _activation(activation),
            nn.Dropout(dropout),
            nn.Linear(channel_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        tokens = self.token_norm(x).transpose(1, 2)
        x = x + self.token_mlp(tokens).transpose(1, 2)
        x = x + self.channel_mlp(self.channel_norm(x))
        return x


class ConvStemMixer(nn.Module):
    """MLP-Mixer style classifier after a convolutional pooling stem."""

    def __init__(self, num_classes=10, widths=(48, 96, 192), blocks_per_stage=2,
                 activation='relu', dropout=0.2):
        super().__init__()
        embed_dim = widths[-1]
        num_tokens = 8 * 8
        token_dim = max(32, widths[0])
        channel_dim = max(embed_dim * 2, widths[1] * 2)
        self.tokenizer = _conv_stem(embed_dim, activation)
        self.blocks = nn.Sequential(*[
            MixerBlock(num_tokens, embed_dim, token_dim, channel_dim, activation, dropout)
            for _ in range(max(1, blocks_per_stage))
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        x = self.tokenizer(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.blocks(x)
        return self.head(x.mean(dim=1))
