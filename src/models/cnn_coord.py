"""Small CNN that keeps position information: CoordConv input + flattened 8x8 grid (no global pooling).

    image (1, 128, 128) + x/y coordinate maps          3 x 128 x 128
    Block 1  Conv 5x5 (3->32)               BN ReLU AvgPool 2    32 x 64 x 64   smooths sparse sim hits
    Block 2  [Conv 3x3 BN ReLU] x 2 (->64)          MaxPool 2    64 x 32 x 32
    Block 3  [Conv 3x3 BN ReLU] x 2 (->128)         MaxPool 2   128 x 16 x 16
    Block 4  [Conv 3x3 BN ReLU] x 2 (->128)         MaxPool 2   128 x  8 x  8
    head="flatten":  Conv 1x1 (->32) BN ReLU, Flatten           2048   (where each feature is, is kept)
    head="avgpool":  global average pooling                      128   (like the ConvMixer: position lost)
    Linear -> BN -> ReLU -> Dropout (256), Linear -> BN -> ReLU (64) = features(x), Linear -> X, Y, Z

CoordConv: two extra input channels with the x and y coordinate of each pixel (-1 .. 1), so filters can
respond differently depending on where they are in the image. The labels are mostly encoded in the
position and shape of the spot, so both CoordConv and the flattened grid keep that information.
Switches in CONFIG for ablations: coord on/off, head "flatten"/"avgpool", channel widths, dropout.
The input size is fixed in CONFIG ("in_size", must be divisible by 16).
"""
import torch
import torch.nn as nn


def conv_bn_relu(c_in, c_out, k):
    return [nn.Conv2d(c_in, c_out, k, padding=k // 2, bias=False), nn.BatchNorm2d(c_out), nn.ReLU()]


class CNNCoord(nn.Module):
    def __init__(self, in_size=128, coord=True, channels=(32, 64, 128, 128), head="flatten",
                 reduce=32, hidden=(256, 64), dropout=0.2):
        super().__init__()
        if in_size % 16:
            raise ValueError(f"in_size must be divisible by 16 (4 x pooling by 2), not {in_size}")
        if head not in ("flatten", "avgpool"):
            raise ValueError(f"head must be 'flatten' or 'avgpool', not {head!r}")
        self.coord = coord
        if coord:   # (1, 2, H, W): x varies along the columns, y along the rows; moves with .to(device)
            lin = torch.linspace(-1, 1, in_size)
            yy, xx = torch.meshgrid(lin, lin, indexing="ij")
            self.register_buffer("coords", torch.stack([xx, yy])[None], persistent=False)

        c0, c1, c2, c3 = channels
        layers = conv_bn_relu(1 + 2 * coord, c0, 5) + [nn.AvgPool2d(2)]                     # block 1
        for c_in, c_out in ((c0, c1), (c1, c2), (c2, c3)):                                  # blocks 2-4
            layers += conv_bn_relu(c_in, c_out, 3) + conv_bn_relu(c_out, c_out, 3) + [nn.MaxPool2d(2)]
        if head == "flatten":
            layers += conv_bn_relu(c3, reduce, 1) + [nn.Flatten()]
            d = reduce * (in_size // 16) ** 2
        else:
            layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten()]
            d = c3
        for i, n in enumerate(hidden):
            layers += [nn.Linear(d, n), nn.BatchNorm1d(n), nn.ReLU()]
            if dropout > 0 and i < len(hidden) - 1:   # no dropout directly on the features
                layers.append(nn.Dropout(dropout))
            d = n
        self.body = nn.Sequential(*layers)   # feature extractor
        self.head = nn.Linear(d, 3)          # regressor: no activation

    def features(self, x):
        """(B, 1, H, W) -> (B, hidden[-1]): activations of the last hidden layer."""
        if self.coord:
            x = torch.cat([x, self.coords.expand(len(x), -1, -1, -1).to(x.dtype)], dim=1)
        return self.body(x)

    def forward(self, x):
        return self.head(self.features(x))


# sizes used by train.py (--model cnn_coord); logged in each run's config.json
CONFIG = dict(in_size=128, coord=True, channels=(32, 64, 128, 128), head="flatten",
              reduce=32, hidden=(256, 64), dropout=0.2)


def build():
    return CNNCoord(**CONFIG)
