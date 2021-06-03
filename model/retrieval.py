from torch import nn


class Unfold2D(nn.Module):

    def __init__(self, patch_extent, nf):
        super().__init__()
        self.patch_extent = patch_extent
        self.nf = nf

    def forward(self, x):
        unfold_out = x.unfold(2, self.patch_extent, self.patch_extent).unfold(3, self.patch_extent, self.patch_extent)
        return unfold_out.permute((0, 2, 3, 1, 4, 5)).reshape((-1, self.nf, self.patch_extent, self.patch_extent))


class Double3x3Conv2d(nn.Module):

    def __init__(self, nf_in, nf_out):
        super().__init__()
        # noinspection PyTypeChecker
        self.model = nn.Sequential(
            nn.Conv2d(nf_in, nf_out, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf_out, nf_out, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x):
        return self.model(x)


class Patch16(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        # noinspection PyTypeChecker
        self.backbone = nn.Sequential(
            Double3x3Conv2d(3, nf),
            Double3x3Conv2d(nf, nf * 2),
            Double3x3Conv2d(nf * 2, nf * 4),
            Double3x3Conv2d(nf * 4, nf * 8),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.unfold = Unfold2D(1, nf * 8)
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward(self, x):
        volume = self.backbone(x)
        unfolded = self.unfold(volume).squeeze(-1).squeeze(-1)
        return self.final_layer(unfolded)


class FullTexture(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        # noinspection PyTypeChecker
        self.backbone = nn.Sequential(
            Double3x3Conv2d(3, nf),
            Double3x3Conv2d(nf, nf * 2),
            Double3x3Conv2d(nf * 2, nf * 4),
            Double3x3Conv2d(nf * 4, nf * 8),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.final_layer = nn.Linear(8 * nf * 2 * 2, z_dim)

    def forward(self, x):
        volume = self.backbone(x)
        return self.final_layer(volume.view(volume.shape[0], -1))


class Patch16Thin(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        # noinspection PyTypeChecker
        self.backbone = nn.Sequential(
            Double3x3Conv2d(3, nf),
            Double3x3Conv2d(nf, nf * 2),
            Double3x3Conv2d(nf * 2, nf * 4),
            Double3x3Conv2d(nf * 4, nf * 8),
        )
        self.unfold = Unfold2D(1, nf * 8)
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward(self, x):
        volume = self.backbone(x)
        unfolded = self.unfold(volume).squeeze(-1).squeeze(-1)
        return self.final_layer(unfolded)


class Patch16MLP(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(3 * 16**2, nf * 4),
            nn.ReLU(),
            nn.Linear(nf * 4, nf * 8),
            nn.ReLU(),
            nn.Linear(nf * 8, nf * 16),
            nn.ReLU(),
            nn.Linear(nf * 16, nf * 8),
            nn.ReLU(),
            nn.Linear(nf * 8, z_dim),
        ])

    def forward(self, x):
        x = x.reshape([x.shape[0], -1])
        for f in self.layers:
            x = f(x)
        return x


def get_input_feature_extractor(config):
    if config.dictionary.patch_size == 16:
        return Patch16(config.fenc_nf, config.fenc_zdim)
    elif config.dictionary.patch_size == 128:
        return FullTexture(config.fenc_nf, config.fenc_zdim)
    raise NotImplementedError


def get_target_feature_extractor(config):
    if config.dictionary.patch_size == 16:
        return Patch16MLP(config.fenc_nf, config.fenc_zdim)
    elif config.dictionary.patch_size == 128:
        return FullTexture(config.fenc_nf, config.fenc_zdim)
    raise NotImplementedError
