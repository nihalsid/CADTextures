from torch import nn
import torch


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


class Double3x3Conv2dMaxPool(nn.Module):

    def __init__(self, nf_in, nf_out):
        super().__init__()
        # noinspection PyTypeChecker
        self.model = nn.Sequential(
            nn.Conv2d(nf_in, nf_out, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(nf_out, nf_out, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool2d(2),
        )

    def forward(self, x):
        return self.model(x)


class Double3x3MixConv2d(nn.Module):
    def __init__(self, nf_in, nf_out, mask_pool):

        super().__init__()
        self.mask_pool = mask_pool
        # noinspection PyTypeChecker
        self.conv1 = nn.Sequential(
            nn.Conv2d(nf_in, nf_out, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )
        # noinspection PyTypeChecker
        self.conv2 = nn.Sequential(
            nn.Conv2d(nf_out + nf_out, nf_out, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, x, mask_missing):
        mask_missing = (self.mask_pool(mask_missing) > 0).float().detach()
        x = self.conv1(x)
        expanded_mask = (1 - mask_missing).expand(-1, x.shape[1], -1, -1)
        x_valid = x * expanded_mask
        max_value = x_valid.max(dim=-1)[0].max(dim=-1)[0].view((x.shape[0], x.shape[1], 1, 1)).expand(-1, -1, x.shape[2], -1).expand(-1, -1, -1, x.shape[3])
        return self.conv2(torch.cat([x, max_value], 1))


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

    def forward(self, x, _mask):
        volume = self.backbone(x)
        unfolded = self.unfold(volume).squeeze(-1).squeeze(-1)
        return self.final_layer(unfolded)

    def forward_with_attention(self, x, _mask):
        return self.forward(x, _mask), None


class Patch16MaxPool(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        # noinspection PyTypeChecker
        self.backbone = nn.Sequential(
            Double3x3Conv2dMaxPool(3, nf),
            Double3x3Conv2dMaxPool(nf, nf * 2),
            Double3x3Conv2dMaxPool(nf * 2, nf * 4),
            Double3x3Conv2dMaxPool(nf * 4, nf * 8),
            nn.Conv2d(nf * 8, nf * 8, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.unfold = Unfold2D(1, nf * 8)
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward(self, x, _mask):
        volume = self.backbone(x)
        unfolded = self.unfold(volume).squeeze(-1).squeeze(-1)
        return self.final_layer(unfolded)

    def forward_with_attention(self, x, _mask):
        return self.forward(x, _mask), None


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


class SelfAttentionEncoder16(nn.Module):

    backbone_output_height = 8
    backbone_output_width = 8

    def __init__(self, nf, z_dim):
        super().__init__()
        # noinspection PyTypeChecker
        self.backbone = nn.Sequential(
            Double3x3Conv2d(3, nf),
            Double3x3Conv2d(nf, nf * 2),
            Double3x3Conv2d(nf * 2, nf * 4),
            Double3x3Conv2d(nf * 4, nf * 4)
        )
        self.backbone_mask = nn.Sequential(
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2)
        )
        self.attention = nn.MultiheadAttention(embed_dim=nf * 4, num_heads=1, batch_first=True)
        # noinspection PyTypeChecker
        self.post_attention = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 8, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.unfold = Unfold2D(1, nf * 8)
        self.final_layer = nn.Linear(8 * nf, z_dim)

    @staticmethod
    def view_as_image(x):
        b, c = x.shape[0], x.shape[2]
        return x.permute((0, 2, 1)).reshape((b, c, SelfAttentionEncoder16.backbone_output_height, SelfAttentionEncoder16.backbone_output_width))

    @staticmethod
    def view_as_sequence(x):
        b, c = x.shape[0:2]
        return x.permute((0, 2, 3, 1)).reshape((b, -1, c))

    def forward_with_attention(self, x, mask_missing):
        volume = self.backbone(x)
        kp_mask = self.view_as_sequence((self.backbone_mask(mask_missing) > 0).bool().detach()).squeeze(-1)
        volume_as_sequence = self.view_as_sequence(volume)
        attn_output, attn_weights = self.attention(volume_as_sequence, volume_as_sequence, volume_as_sequence, key_padding_mask=kp_mask)
        attn_output = self.view_as_image(attn_output)
        post_attn = self.post_attention(torch.cat((volume, attn_output), 1))
        unfolded = self.unfold(post_attn).squeeze(-1).squeeze(-1)
        return self.final_layer(unfolded), attn_weights

    def forward(self, x, mask_missing):
        return self.forward_with_attention(x, mask_missing)[0]


class MaxMixingEncoder16(nn.Module):

    def __init__(self, nf, z_dim):
        super().__init__()
        backbone_mask_2 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
        )
        backbone_mask_3 = nn.Sequential(
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
            nn.MaxPool2d(2),
        )
        # noinspection PyTypeChecker
        self.backbone_0 = nn.Sequential(
            Double3x3Conv2d(3, nf),
            Double3x3Conv2d(nf, nf * 2)
        )

        self.mix_0 = Double3x3MixConv2d(nf * 2, nf * 4, backbone_mask_2)
        self.mix_1 = Double3x3MixConv2d(nf * 4, nf * 8, backbone_mask_3)

        # noinspection PyTypeChecker
        self.backbone_1 = nn.Sequential(
            nn.Conv2d(nf * 8, nf * 8, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.unfold = Unfold2D(1, nf * 8)
        self.final_layer = nn.Linear(8 * nf, z_dim)

    def forward_with_attention(self, x, mask_missing):
        volume_0 = self.backbone_0(x)
        mix_0 = self.mix_0(volume_0, mask_missing)
        mix_1 = self.mix_1(mix_0, mask_missing)
        volume = self.backbone_1(mix_1)
        unfolded = self.unfold(volume).squeeze(-1).squeeze(-1)
        return self.final_layer(unfolded), None

    def forward(self, x, mask_missing):
        return self.forward_with_attention(x, mask_missing)[0]


def get_input_feature_extractor(config):
    if config.dictionary.patch_size == 16:
        if config.dictionary.extractor == "self_attention":
            return SelfAttentionEncoder16(config.fenc_nf, config.fenc_zdim)
        if config.dictionary.extractor == "max_mixing":
            return MaxMixingEncoder16(config.fenc_nf, config.fenc_zdim)
        if config.dictionary.extractor == "max_pool":
            return Patch16MaxPool(config.fenc_nf, config.fenc_zdim)
        else:
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
