from torch import nn

from model.scribbler import ResidualBlock


class Discriminator(nn.Module):

    def __init__(self, input_nc, ndf, use_sigmoid):
        super(Discriminator, self).__init__()
        self.input_nc = input_nc

        # noinspection PyTypeChecker
        sequence = [
            nn.Conv2d(self.input_nc, ndf, kernel_size=9, stride=2, padding=0),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf, ndf * 2, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(ndf * 2, ndf * 8, kernel_size=5, stride=2, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Dropout(0.2),

            ResidualBlock(ndf * 8, ndf * 8),
            ResidualBlock(ndf * 8, ndf * 8),

            nn.Conv2d(ndf * 8, ndf * 4, kernel_size=4, stride=2, padding=0),
            nn.Dropout(0.2),

            nn.Conv2d(ndf * 4, 1, kernel_size=4, stride=2, padding=0)
        ]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        else:
            sequence += [nn.ReLU(inplace=True)]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        return self.model(x)
