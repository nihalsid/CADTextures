import torch
import torch.nn as nn


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1, downsample=None, dilation=(1, 1), residual=True, conv=nn.Conv2d, bn=lambda c: nn.GroupNorm(4, c)):
        super().__init__()
        # noinspection PyTypeChecker
        self.conv1 = conv(in_channels, out_channels, kernel_size=3, padding=dilation[0], dilation=dilation[0])
        self.bn1 = bn(out_channels)
        self.relu = nn.ReLU(inplace=True)
        # noinspection PyTypeChecker
        self.conv2 = conv(out_channels, out_channels, kernel_size=3, padding=dilation[0], dilation=dilation[0])
        self.bn2 = bn(out_channels)
        self.downsample = downsample
        self.stride = stride
        self.residual = residual

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        if self.residual:
            out += residual

        out = self.relu(out)
        return out


class UpsamplingBlock(nn.Module):

    def __init__(self, input_nc, output_nc, kernel, stride, pad):
        super().__init__()
        self.conv = nn.Conv2d(input_nc, output_nc, kernel, stride, pad)

    def forward(self, x):
        out = self.conv(x)
        return nn.functional.interpolate(out, mode='bilinear', scale_factor=2, align_corners=True)


class Scribbler(nn.Module):

    def __init__(self, input_nc, output_nc, ngf):
        super().__init__()
        norm = lambda c: nn.GroupNorm(4, c)
        # noinspection PyTypeChecker
        self.module_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_nc, ngf, 3, 1, 1),
                norm(ngf),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf, ngf),
                nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
                norm(ngf * 2),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf * 2, ngf * 2),
                nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
                norm(ngf * 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf * 4, ngf * 4),
                nn.Conv2d(ngf * 4, ngf * 8, 3, 2, 1),
                norm(ngf * 8),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf * 8, ngf * 8),
                ResidualBlock(ngf * 8, ngf * 8),
                ResidualBlock(ngf * 8, ngf * 8),
                ResidualBlock(ngf * 8, ngf * 8),
                ResidualBlock(ngf * 8, ngf * 8),
            ),
            nn.Sequential(
                UpsamplingBlock(ngf * 8, ngf * 4, 3, 1, 1),
                norm(ngf * 4),
                nn.ReLU(True),
                ResidualBlock(ngf * 4, ngf * 4),
                ResidualBlock(ngf * 4, ngf * 4),
            ),
            nn.Sequential(
                UpsamplingBlock(ngf * 4, ngf * 2, 3, 1, 1),
                norm(ngf * 2),
                nn.ReLU(True),
                ResidualBlock(ngf * 2, ngf * 2),
                ResidualBlock(ngf * 2, ngf * 2),
            ),
            nn.Sequential(
                UpsamplingBlock(ngf * 2, ngf, 3, 1, 1),
                norm(ngf),
                nn.ReLU(True),
                ResidualBlock(ngf, ngf),
                norm(ngf),
                ResidualBlock(ngf, ngf),
            ),
            nn.Conv2d(ngf, 3, output_nc, 1, 1)
        ])

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x


class ImageFusionScribbler(nn.Module):

    def __init__(self, input_nc, input_nc_image, output_nc, ngf, ngf_image):
        super().__init__()
        norm = lambda c: nn.GroupNorm(4, c)
        # noinspection PyTypeChecker
        self.image_features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_nc_image, ngf_image, 3, 1, 1),
                norm(ngf_image),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(ngf_image, ngf_image, 3, 2, 1),
                norm(ngf_image),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf_image, ngf_image),
                nn.Conv2d(ngf_image, ngf_image * 2, 3, 2, 1),
                norm(ngf_image * 2),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf_image * 2, ngf_image * 2),
                nn.Conv2d(ngf_image * 2, ngf_image * 4, 3, 2, 1),
                norm(ngf_image * 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf_image * 4, ngf_image * 4),
                nn.Conv2d(ngf_image * 4, ngf_image * 4, 3, 2, 1),
                norm(ngf_image * 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf_image * 4, ngf_image * 4),
                nn.Conv2d(ngf_image * 4, ngf_image * 8, 3, 2, 1),
                norm(ngf_image * 8),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf_image * 8, ngf_image * 8),
                nn.Conv2d(ngf_image * 8, ngf_image * 8, 3, 2, 1),
                norm(ngf_image * 8),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(ngf_image * 8, ngf_image * 8, 4, 1, 0),
                norm(ngf_image * 8),
                nn.ReLU(True),
            )
        ])
        # noinspection PyTypeChecker
        self.map_features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_nc, ngf, 3, 1, 1),
                norm(ngf),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf, ngf),
                nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
                norm(ngf * 2),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf * 2, ngf * 2),
                nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
                norm(ngf * 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf * 4, ngf * 4),
                nn.Conv2d(ngf * 4, ngf * 8, 3, 2, 1),
                norm(ngf * 8),
                nn.ReLU(True),
            ),
        ])
        # noinspection PyTypeChecker
        self.fusion = nn.Sequential(
            nn.Conv2d(ngf * 8 + ngf_image * 8, ngf * 8, 3, 1, 1),
            norm(ngf * 8),
            nn.ReLU(True),
        )
        # noinspection PyTypeChecker
        self.decoder = nn.ModuleList([
            nn.Sequential(
                ResidualBlock(ngf * 8, ngf * 8),
                ResidualBlock(ngf * 8, ngf * 8),
                ResidualBlock(ngf * 8, ngf * 8),
                ResidualBlock(ngf * 8, ngf * 8),
            ),
            nn.Sequential(
                UpsamplingBlock(ngf * 8, ngf * 4, 3, 1, 1),
                norm(ngf * 4),
                nn.ReLU(True),
                ResidualBlock(ngf * 4, ngf * 4),
                ResidualBlock(ngf * 4, ngf * 4),
            ),
            nn.Sequential(
                UpsamplingBlock(ngf * 4, ngf * 2, 3, 1, 1),
                norm(ngf * 2),
                nn.ReLU(True),
                ResidualBlock(ngf * 2, ngf * 2),
                ResidualBlock(ngf * 2, ngf * 2),
            ),
            nn.Sequential(
                UpsamplingBlock(ngf * 2, ngf, 3, 1, 1),
                norm(ngf),
                nn.ReLU(True),
                ResidualBlock(ngf, ngf),
                norm(ngf),
                ResidualBlock(ngf, ngf),
            ),
            nn.Conv2d(ngf, 3, output_nc, 1, 1)
        ])
        self.tanh = nn.Tanh()

    def fuse_features(self, input_maps, *condition):
        image = condition[0]
        map_features = nn.Sequential(*self.map_features)(input_maps)
        image_features = nn.Sequential(*self.image_features)(image)
        x = torch.cat([map_features, image_features.expand(-1, -1, map_features.shape[-2], -1).expand(-1, -1, -1, map_features.shape[-1])], dim=1)
        return self.fusion(x)
    
    def forward(self, input_maps, *condition):
        image = condition[0]
        x = self.fuse_features(input_maps, image)
        for module in self.decoder:
            x = module(x)
        return self.tanh(x) * 0.5


class ImageFusionScribblerSlim(nn.Module):

    def __init__(self, input_nc, input_nc_image, output_nc, ngf, ngf_image):
        super().__init__()
        norm = lambda c: nn.GroupNorm(4, c)
        # noinspection PyTypeChecker
        self.image_features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_nc_image, ngf_image, 3, 2, 1),
                norm(ngf_image),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(ngf_image, ngf_image * 2, 3, 2, 1),
                norm(ngf_image * 2),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(ngf_image * 2, ngf_image * 4, 3, 2, 1),
                norm(ngf_image * 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(ngf_image * 4, ngf_image * 4, 3, 2, 1),
                norm(ngf_image * 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(ngf_image * 4, ngf_image * 8, 3, 2, 1),
                norm(ngf_image * 8),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(ngf_image * 8, ngf_image * 8, 3, 2, 1),
                norm(ngf_image * 8),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(ngf_image * 8, ngf_image * 8, 4, 1, 0),
                norm(ngf_image * 8),
                nn.ReLU(True),
            )
        ])
        # noinspection PyTypeChecker
        self.map_features = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_nc, ngf, 3, 1, 1),
                norm(ngf),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
                norm(ngf * 2),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
                norm(ngf * 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Conv2d(ngf * 4, ngf * 8, 3, 2, 1),
                norm(ngf * 8),
                nn.ReLU(True),
            ),
        ])
        # noinspection PyTypeChecker
        self.fusion = nn.Sequential(
            nn.Conv2d(ngf * 8 + ngf_image * 8, ngf * 8, 3, 1, 1),
            norm(ngf * 8),
            nn.ReLU(True),
        )
        # noinspection PyTypeChecker
        self.decoder = nn.ModuleList([
            nn.Sequential(
                UpsamplingBlock(ngf * 8, ngf * 4, 3, 1, 1),
                norm(ngf * 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                UpsamplingBlock(ngf * 4, ngf * 2, 3, 1, 1),
                norm(ngf * 2),
                nn.ReLU(True),
            ),
            nn.Sequential(
                UpsamplingBlock(ngf * 2, ngf, 3, 1, 1),
                norm(ngf),
                nn.ReLU(True),
            ),
            nn.Conv2d(ngf, 3, output_nc, 1, 1)
        ])
        self.tanh = nn.Tanh()

    def fuse_features(self, input_maps, *condition):
        image = condition[0]
        map_features = nn.Sequential(*self.map_features)(input_maps)
        image_features = nn.Sequential(*self.image_features)(image)
        x = torch.cat([map_features, image_features.expand(-1, -1, map_features.shape[-2], -1).expand(-1, -1, -1, map_features.shape[-1])], dim=1)
        return self.fusion(x)

    def forward(self, input_maps, *condition):
        image = condition[0]
        x = self.fuse_features(input_maps, image)
        for module in self.decoder:
            x = module(x)
        return self.tanh(x) * 0.5


class ImageAnd3dFusionScribbler(ImageFusionScribbler):

    def __init__(self, input_nc, input_nc_image, output_nc, ngf, ngf_image, ngf_3d):
        super().__init__(input_nc, input_nc_image, output_nc, ngf, ngf_image)
        norm = lambda c: nn.GroupNorm(4, c)
        # noinspection PyTypeChecker
        self.df_features = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(1, ngf_3d, 3, 1, 1),
                norm(ngf_3d),
                nn.ReLU(True),
            ),
            nn.Sequential(
                nn.Conv3d(ngf_3d, ngf_3d * 2, 3, 2, 1),
                norm(ngf_3d * 2),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf_3d * 2, ngf_3d * 2, conv=nn.Conv3d, bn=norm),
                nn.Conv3d(ngf_3d * 2, ngf_3d * 4, 3, 2, 1),
                norm(ngf_3d * 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf_3d * 4, ngf_3d * 4, conv=nn.Conv3d, bn=norm),
                nn.Conv3d(ngf_3d * 4, ngf_3d * 4, 3, 2, 1),
                norm(ngf_3d * 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf_3d * 4, ngf_3d * 4, conv=nn.Conv3d, bn=norm),
                nn.Conv3d(ngf_3d * 4, ngf_3d * 8, 3, 2, 1),
                norm(ngf_3d * 8),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf_3d * 8, ngf_3d * 8, conv=nn.Conv3d, bn=norm),
                nn.Conv3d(ngf_3d * 8, ngf_3d * 8, 4, 1, 0),
                norm(ngf_3d * 8),
                nn.ReLU(True),
            ),
        ])
        # noinspection PyTypeChecker
        self.fusion = nn.Sequential(
            nn.Conv2d(ngf * 8 + ngf_image * 8 + ngf_3d * 8, ngf * 8, 3, 1, 1),
            norm(ngf * 8),
            nn.ReLU(True),
        )

    def fuse_features(self, input_maps, *condition):
        image = condition[0]
        distance_field = condition[1]
        map_features = nn.Sequential(*self.map_features)(input_maps)
        image_features = nn.Sequential(*self.image_features)(image)
        df_features = nn.Sequential(*self.df_features)(distance_field).squeeze(4)
        x = torch.cat([map_features,
                       image_features.expand(-1, -1, map_features.shape[-2], -1).expand(-1, -1, -1, map_features.shape[-1]),
                       df_features.expand(-1, -1, map_features.shape[-2], -1).expand(-1, -1, -1, map_features.shape[-1])], dim=1)
        return self.fusion(x)

    def forward(self, input_maps, *condition):
        image = condition[0]
        distance_field = condition[1]
        x = self.fuse_features(input_maps, image, distance_field)
        for module in self.decoder:
            x = module(x)
        return self.tanh(x) * 0.5


def get_model(config):
    map_channels = 1
    render_channels = 4
    if 'noc' in config.inputs:
        map_channels += 3
    if 'partial_texture' in config.inputs:
        map_channels += 3
    if 'normal' in config.inputs:
        map_channels += 3
    if 'noc_render' in config.inputs:
        render_channels += 3
    if 'distance_field' in config.inputs:
        if config.model.slim:
            raise NotImplementedError
        return ImageAnd3dFusionScribbler(map_channels, render_channels, 3, config.model.input_texture_ngf, config.model.render_ngf, config.model.df_ngf)
    else:
        if config.model.slim:
            return ImageFusionScribblerSlim(map_channels, render_channels, 3, config.model.input_texture_ngf, config.model.render_ngf)
        return ImageFusionScribbler(map_channels, render_channels, 3, config.model.input_texture_ngf, config.model.render_ngf)
