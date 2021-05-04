from torch import nn

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


def get_discriminator(config):
    return TextureGANDiscriminator(1, config.model.discriminator_ngf)


def get_discriminator_local(config):
    return TextureGANDiscriminatorLocal(2, config.model.discriminator_ngf)
