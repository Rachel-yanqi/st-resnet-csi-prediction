
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

    def forward(self, x):
        return x + self.block(x)

class STResNet(nn.Module):
    def __init__(self, in_channels=2, num_blocks=3, num_users=4):
        super().__init__()
        self.start = nn.Conv2d(in_channels * 5, 64, kernel_size=3, padding=1)  # 5 = history length
        self.res_blocks = nn.Sequential(*[ResidualBlock(64) for _ in range(num_blocks)])
        self.end = nn.Conv2d(64, in_channels, kernel_size=3, padding=1)

    def forward(self, x):  # x: (B, T=5, 2, N, N)
        B, T, C, H, W = x.shape
        x = x.reshape(B, T * C, H, W)  # merge T and C
        x = self.start(x)
        x = self.res_blocks(x)
        return self.end(x)  # (B, 2, H, W)
