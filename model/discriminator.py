import math

import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from model.texture_gan import ResidualBlock


class ConvWithNormAndActivation(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias, norm, activation=nn.LeakyReLU(0.2, inplace=True)):
        super(ConvWithNormAndActivation, self).__init__()
        self.conv = nn.ModuleList()
        self.conv.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, bias=bias, padding=padding))
        if activation is not None:
            self.conv.append(activation)
        if norm == "spectral":
            self.conv[0] = nn.utils.spectral_norm(self.conv[0])
        elif norm == "batch":
            self.conv.append(nn.BatchNorm3d(out_channels))

    def forward(self, x):
        for m in self.conv:
            x = m(x)
        return x


class SanityTestDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf):
        super(SanityTestDiscriminator, self).__init__()
        self.input_nc = input_nc
        # noinspection PyTypeChecker
        layer_lists = [
            ConvWithNormAndActivation(input_nc, ndf, 4, 2, 1, True, 'none'),
            ConvWithNormAndActivation(ndf, ndf * 2, 4, 2, 1, True, 'none'),
            ConvWithNormAndActivation(ndf * 2, ndf * 4, 4, 2, 1, True, 'none'),
            ConvWithNormAndActivation(ndf * 4, ndf * 8, 4, 2, 1, True, 'none'),
            nn.Conv2d(ndf * 8, 1, 4, stride=1, padding=0)
        ]
        self.model = nn.Sequential(*layer_lists)

    def forward(self, x):
        return self.model(x)


class TextureGANDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf):
        super(TextureGANDiscriminator, self).__init__()

        self.input_nc = input_nc
        self.ndf = ndf
        self.res_block = ResidualBlock

        self.model = self.create_discriminator()

    def create_discriminator(self):
        self.res_block = ResidualBlock
        # noinspection PyTypeChecker
        sequence = [
            nn.Conv2d(self.input_nc, self.ndf, kernel_size=9, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf * 2, self.ndf * 8, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.2),

            self.res_block(self.ndf * 8, self.ndf * 8),
            self.res_block(self.ndf * 8, self.ndf * 8),

            nn.Conv2d(self.ndf * 8, self.ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.2),

            nn.Conv2d(self.ndf * 4, 1, kernel_size=4, stride=2, padding=1)
        ]

        return nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class TextureGANDiscriminatorLocal(nn.Module):

    def __init__(self, input_nc, ndf):
        super(TextureGANDiscriminatorLocal, self).__init__()
        # noinspection PyTypeChecker
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ndf),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, ndf * 4, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, True),

            ResidualBlock(ndf * 4, ndf * 4),
            ResidualBlock(ndf * 4, ndf * 4),

            nn.Conv2d(ndf * 4, ndf * 2, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 2, 1, kernel_size=3, stride=2, padding=1)
        ]
        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


class TextureGANDiscriminatorSlim(nn.Module):

    def __init__(self, input_nc, ndf):
        super(TextureGANDiscriminatorSlim, self).__init__()
        self.input_nc = input_nc
        self.ndf = ndf
        self.res_block = ResidualBlock
        self.model = self.create_discriminator()

    def create_discriminator(self):
        self.res_block = ResidualBlock
        # noinspection PyTypeChecker
        sequence = [
            nn.Conv2d(self.input_nc, self.ndf, kernel_size=9, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf, self.ndf * 2, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(self.ndf * 2, self.ndf * 8, kernel_size=5, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.2),

            nn.Conv2d(self.ndf * 8, self.ndf * 4, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.2),

            nn.Conv2d(self.ndf * 4, 1, kernel_size=4, stride=2, padding=1)
        ]

        return nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)


def kaiming_leaky_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        torch.nn.init.kaiming_normal_(m.weight, a=0.2, mode='fan_in', nonlinearity='leaky_relu')


class CoordConv(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, in_channels, out_channels, with_r=False, **kwargs):
        super().__init__()
        self.addcoords = AddCoords(with_r=with_r)
        in_size = in_channels + 2
        if with_r:
            in_size += 1
        self.conv = nn.Conv2d(in_size, out_channels, **kwargs)

    def forward(self, x):
        ret = self.addcoords(x)
        ret = self.conv(ret)
        return ret


class AddCoords(nn.Module):
    """
    Source: https://github.com/mkocabas/CoordConv-pytorch/blob/master/CoordConv.py
    """

    def __init__(self, with_r=False):
        super().__init__()
        self.with_r = with_r

    def forward(self, input_tensor):
        """
        Args:
            input_tensor: shape(batch, channel, x_dim, y_dim)
        """
        batch_size, _, x_dim, y_dim = input_tensor.size()

        xx_channel = torch.arange(x_dim).repeat(1, y_dim, 1)
        yy_channel = torch.arange(y_dim).repeat(1, x_dim, 1).transpose(1, 2)

        xx_channel = xx_channel.float() / (x_dim - 1)
        yy_channel = yy_channel.float() / (y_dim - 1)

        xx_channel = xx_channel * 2 - 1
        yy_channel = yy_channel * 2 - 1

        xx_channel = xx_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)
        yy_channel = yy_channel.repeat(batch_size, 1, 1, 1).transpose(2, 3)

        ret = torch.cat([
            input_tensor,
            xx_channel.type_as(input_tensor),
            yy_channel.type_as(input_tensor)], dim=1)

        if self.with_r:
            rr = torch.sqrt(torch.pow(xx_channel.type_as(input_tensor) - 0.5, 2) + torch.pow(yy_channel.type_as(input_tensor) - 0.5, 2))
            ret = torch.cat([ret, rr], dim=1)

        return ret


class ResidualCCBlock(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3):
        super().__init__()
        p = kernel_size // 2
        self.network = nn.Sequential(
            CoordConv(inplanes, planes, kernel_size=kernel_size, padding=p),
            nn.LeakyReLU(0.2, inplace=True),
            CoordConv(planes, planes, kernel_size=kernel_size, stride=2, padding=p),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.network.apply(kaiming_leaky_init)

        self.proj = nn.Conv2d(inplanes, planes, 1, stride=2)

    def forward(self, input):
        y = self.network(input)

        identity = self.proj(input)

        y = (y + identity) / math.sqrt(2)
        return y


class AdapterBlock(nn.Module):
    def __init__(self, output_channels):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, output_channels, 1, padding=0),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, input):
        return self.model(input)


class CCSDiscriminator(nn.Module):

    def __init__(self, **kwargs):  # from 4 * 2^0 to 4 * 2^7 4 -> 512
        super().__init__()
        self.epoch = 0
        self.step = 0
        self.layers = nn.ModuleList(
            [
                ResidualCCBlock(32, 64),  # 6 256x256 -> 128x128
                ResidualCCBlock(64, 128),  # 5 128x128 -> 64x64
                ResidualCCBlock(128, 256),  # 4 64x64 -> 32x32
                ResidualCCBlock(256, 400),  # 3 32x32 -> 16x16
                ResidualCCBlock(400, 400),  # 2 16x16 -> 8x8
                ResidualCCBlock(400, 400),  # 1 8x8 -> 4x4
                ResidualCCBlock(400, 400),  # 7 4x4 -> 2x2
            ])

        self.fromRGB = nn.ModuleList(
            [
                AdapterBlock(32),
                AdapterBlock(64),
                AdapterBlock(128),
                AdapterBlock(256),
                AdapterBlock(400),
                AdapterBlock(400),
                AdapterBlock(400),
                AdapterBlock(400)
            ])
        self.final_layer = nn.Conv2d(400, 1, 2)
        self.img_size_to_layer = {2: 7, 4: 6, 8: 5, 16: 4, 32: 3, 64: 2, 128: 1, 256: 0}

        self.pose_layer = nn.Linear(2, 400)

    def forward(self, input, alpha=1, **kwargs):
        start = self.img_size_to_layer[input.shape[-1]]
        x = self.fromRGB[start](input)

        if kwargs.get('instance_noise', 0) > 0:
            x = x + torch.randn_like(x) * kwargs['instance_noise']

        for i, layer in enumerate(self.layers[start:]):
            if i == 1 and alpha < 1:
                x = alpha * x + (1 - alpha) * self.fromRGB[start + 1](F.interpolate(input, scale_factor=0.5, mode='nearest'))

            x = layer(x)

        x = self.final_layer(x).reshape(x.shape[0], 1)

        return x


def get_discriminator(config):
    return TextureGANDiscriminator(1, config.model.discriminator_ngf)


def get_discriminator_local(config):
    return TextureGANDiscriminatorLocal(2, config.model.discriminator_ngf)


if __name__ == "__main__":
    model = CCSDiscriminator()
    t_in = torch.zeros([4, 3, 128, 128]).float()
    print(model(t_in).shape)
