"""Fully connected network on all pixels, adapted from the best model of Project 1 (FCNet, "wide").

    image (1, 128, 128) -> Flatten (16384) -> [Linear -> BatchNorm1d -> ReLU] x 3 -> Linear -> X, Y, Z

Differences to Project 1: 3 outputs instead of 2; no shift augmentation (here the spot position
carries the labels); trained with the same settings as every model in train.py.
The input size is fixed in CONFIG: change "in_size" when training on a downsampled cache.
features(x) = activations of the last hidden layer (t-SNE, domain discriminator).
"""
import torch.nn as nn


class FCNet(nn.Module):
    def __init__(self, in_size=128, hidden=(2048, 1024, 256), dropout=0.0):
        super().__init__()
        layers, d = [nn.Flatten()], in_size * in_size
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


# sizes used by train.py (--model fc); logged in each run's config.json
CONFIG = dict(in_size=128, hidden=(2048, 1024, 256), dropout=0.0)


def build():
    return FCNet(**CONFIG)
