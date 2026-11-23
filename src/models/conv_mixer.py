from torch import nn


class ConvMixer(nn.Module):
    def __init__(self, patch_size, embedding_dim, num_blocks, kernel_size):
        super().__init__()
        self.num_blocks = num_blocks

        self.embedding = nn.Conv2d(1, embedding_dim, patch_size, patch_size)

        self.depthwise = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm2d(embedding_dim),
                    nn.Conv2d(
                        embedding_dim,
                        embedding_dim,
                        kernel_size,
                        groups=embedding_dim,
                        padding="same",
                    ),
                    nn.GELU(),
                )
                for _ in range(num_blocks)
            ]
        )

        self.pointwise = nn.ModuleList(
            [
                nn.Sequential(
                    nn.BatchNorm2d(embedding_dim),
                    nn.Conv2d(embedding_dim, embedding_dim, 1),
                    nn.GELU(),
                )
                for _ in range(num_blocks)
            ]
        )

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(embedding_dim, 3)
        )

    def forward(self, x):
        x = self.embedding(x)
        for dw, pw in zip(self.depthwise, self.pointwise):
            x = dw(x) + x
            x = pw(x)
        return self.head(x)
