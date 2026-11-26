"""Fully connected network on row/column profiles instead of all pixels.

    image (1, 128, 128) -> mean per row (128) + mean per column (128) = 256 values
                        -> [Linear -> BatchNorm1d -> ReLU] x 3 -> Linear -> X, Y, Z

The beam spot's position and width show up directly in these curves, so the network gets
far fewer inputs (256 instead of 16384) with most of the information. Same hidden layers as
fc.py, so the two models differ only in the input.
The input size is fixed in CONFIG: change "in_size" when training on a downsampled cache.
features(x) = activations of the last hidden layer (t-SNE, domain discriminator).
"""
import torch
import torch.nn as nn


class Profiles(nn.Module):
    """(B, 1, H, W) -> (B, H + W): mean over each row, then mean over each column."""

    def forward(self, x):
        x = x[:, 0]
        return torch.cat([x.mean(2), x.mean(1)], dim=1)


class FCProfiles(nn.Module):
    def __init__(self, in_size=128, hidden=(2048, 1024, 256), dropout=0.0):
        super().__init__()
        layers, d = [Profiles()], 2 * in_size
        for n in hidden:
            layers += [nn.Linear(d, n), nn.BatchNorm1d(n), nn.ReLU()]
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            d = n
        self.body = nn.Sequential(*layers)   # feature extractor
        self.head = nn.Linear(d, 3)          # regressor: no activation

    def features(self, x):
        """(B, 1, H, W) -> (B, hidden[-1]): activations of the last hidden layer."""
        return self.body(x)

    def forward(self, x):
        return self.head(self.features(x))


# sizes used by train.py (--model fc_profiles); logged in each run's config.json
CONFIG = dict(in_size=128, hidden=(2048, 1024, 256), dropout=0.0)


def build():
    return FCProfiles(**CONFIG)
