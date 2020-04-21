import torch
import torch.nn as nn
import numpy as np

class ConvNorm(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ConvNorm, self).__init__()

        self.conv = nn.Conv2d(in_channel, out_channel, 3, stride, 1)
        self.norm = nn.BatchNorm2d(out_channel)


    def forward(self, x):
        out = self.norm(self.conv(x))
        return out


class ResBlock(nn.Module):
    def __init__(self, channel):
        super(ResBlock, self).__init__()

        self.channel = channel
        self.convnorm1 = ConvNorm(channel, channel)
        self.convnorm2 = ConvNorm(channel, channel)
        self.prelu = nn.PReLU()

    
    def forward(self, x):
        residual = self.prelu(self.convnorm1(x))
        residual = self.convnorm2(residual)
        return residual + x

    
class Upsample(nn.Module):
    def __init__(self, in_channel, up_factor):
        super(Upsample, self).__init__()

        self.conv = nn.Conv2d(in_channel, in_channel * (up_factor**2), 3, 1, 1)
        self.shuffle = nn.PixelShuffle(up_factor)
        self.prelu = nn.PReLU()

    
    def forward(self, x):
        out = self.prelu(self.shuffle(self.conv(x)))
        return out


class Generator(nn.Module):
    def __init__(self, num_resblock, up_factor):
        super(Generator, self).__init__()
        num_upsample = int(np.log2(up_factor))

        # Pre-resblock
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 9, 1, 4),
            nn.PReLU()
        ) 
        # Resblocks
        self.block2 = nn.Sequential(*[ResBlock(64) for _ in range(num_resblock)])
        # Post resblock
        self.block3 = ConvNorm(64, 64)
        # Upsampling block
        self.block4 = nn.Sequential(*[Upsample(64, 2) for _ in range(num_upsample)])
        self.final_conv = nn.Conv2d(64, 3, 9, 1, 4)


    def forward(self, x):
        tmp = self.block1(x)
        out = tmp
        out = self.block2(out)
        out = self.block3(out) + tmp
        out = self.block4(out)
        out = self.final_conv(out)

        return (torch.tanh(out) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.LeakyReLU(0.2),
            ConvNorm(64, 64, stride=2),
            nn.LeakyReLU(0.2),

            ConvNorm(64, 128),
            nn.LeakyReLU(0.2),
            ConvNorm(128, 128, stride=2),
            nn.LeakyReLU(0.2),

            ConvNorm(128, 256),
            nn.LeakyReLU(0.2),
            ConvNorm(256, 256, stride=2),
            nn.LeakyReLU(0.2),

            ConvNorm(256, 512),
            nn.LeakyReLU(0.2),
            ConvNorm(512, 512, stride=2),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1)
        )


    def forward(self, x):
        batch_size = x.size(0)
        return torch.sigmoid(self.net(x).view(batch_size))
    