import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from omegaconf.dictconfig import DictConfig

from dataset.texture_map_dataset import TextureMapDataset
from model.texture_gan import TextureGAN
from util.regression_loss import RegressionLossHelper

config = {
    'model': {
        'discriminator_ngf': '${model.input_texture_ngf}',
        'input_texture_ngf': 32,
        'render_ngf': 12,
        'df_ngf': 8,
        'slim': False
    },
    'experiment': '01060846_TestShape_Cube_test_run',
    'seed': 139,
    'wandb_main': False,
    'suffix': '-dev',
    'save_epoch': 15,
    'sanity_steps': 1,
    'max_epoch': 10000,
    'val_check_percent': 1.0,
    'val_check_interval': 10,
    'resume': None,
    'batch_size': 4,
    'num_workers': 8,
    'lr': 0.0001,
    'lambda_regr_l': 10,
    'lambda_regr_ab': 10,
    'lambda_content': 0.075,
    'lambda_style': 0.0075,
    'lambda_g': 1,
    'lambda_g_local': 1,
    'lambda_gp': 10,
    'gan_loss_type': 'wgan_gp',
    'regression_loss_type': 'l2',
    'num_patches': 5,
    'patch_size': 25,
    'dataset': {
        'name': 'TestShape/Cube',
        'preload': False,
        'views_per_shape': 12,
        'data_dir': 'data',
        'splits_dir': 'overfit',
        'texture_map_size': 128,
        'render_size': 256,
        'mesh_dir': 'TestShape-model/Cube',
        'color_space': 'rgb'
    },
    'inputs': ['partial_texture']
}
config = DictConfig(config)


def torch_to_np(img_torch):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  W x H x C [0..1]
    '''
    img_np = img_torch.detach().cpu().numpy()[0]
    img_np = np.moveaxis(img_np, 0, 2)
    return img_np


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]


def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False


def get_noise(input_depth, method, spatial_size, noise_type='u', var=1. / 10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`) 
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler. 
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    shape = [1, input_depth, spatial_size[0], spatial_size[1]]

    if method == 'noise':
        net_input = torch.zeros(shape)
        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(
            np.arange(0, spatial_size[1]) / float(spatial_size[1] - 1),
            np.arange(0, spatial_size[0]) / float(spatial_size[0] - 1))
        meshgrid = np.concatenate([X[None, :], Y[None, :]])
        net_input = np_to_torch(meshgrid)
    elif method == 'ones':
        net_input = torch.ones(shape)
    elif method == 'pe':  # positional encoding
        h, w = spatial_size[0], spatial_size[1]
        X = torch.arange(0, w).tile(h, 1)
        Y = torch.arange(0, h).tile(w, 1).T
        tensor_list = []
        for i in range(0, input_depth):
            multiplier = (i + 1) * (2 * np.pi)
            _sin = torch.sin(X * (multiplier / w))
            _cos = torch.cos(Y * (multiplier / h))
            tensor_list.append(_sin)
            tensor_list.append(_cos)
        net_input = torch.stack(tensor_list)
        net_input = net_input.unsqueeze(0)
    else:
        raise NotImplementedError

    return net_input


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


# class Generator(nn.Module):
#     def __init__(self, nz, ngf=64, nc=3):
#         super(Generator, self).__init__()
#         bias = True
#         self.upconv1 = nn.TransposeConv2d(nz, 8 * ngf, 4, 2, 1, bias=bias)
#         self.upconv2 = nn.TransposeConv2d(8 * ngf, 4 * ngf, 4, 2, 1, bias=bias)
#         self.upconv3 = nn.TransposeConv2d(4 * ngf, 2 * ngf, 4, 2, 1, bias=bias)
#         self.upconv4 = nn.TransposeConv2d(2 * ngf, ngf, 4, 2, 1, bias=bias)
#         self.upconv5 = nn.TransposeConv2d(ngf, nc, 4, 2, 1, bias=bias)

#     def forward(self, input):
#         x = self.upconv1(input)
#         x = F.relu(x, inplace=True)
#         x = self.upconv2(x)
#         x = F.relu(x, inplace=True)
#         x = self.upconv3(x)
#         x = F.relu(x, inplace=True)
#         x = self.upconv4(x)
#         x = F.relu(x, inplace=True)
#         x = self.upconv5(x)
#         x = 0.5 * F.tanh(x)
#         return x

# class Generator(nn.Module):
#     def __init__(self, nz, ngf=64, nc=3):
#         super(Generator, self).__init__()
#         bias = True
#         kernel = 5
#         stride = 2
#         padding = (kernel - 1) // 2

#         self.downconv1 = nn.Conv2d(nz,
#                                    ngf,
#                                    kernel,
#                                    stride,
#                                    padding,
#                                    padding_mode="reflect",
#                                    bias=bias)
#         self.downconv2 = nn.Conv2d(ngf,
#                                    2 * ngf,
#                                    kernel,
#                                    stride,
#                                    padding,
#                                    padding_mode="reflect",
#                                    bias=bias)
#         self.downconv3 = nn.Conv2d(2 * ngf,
#                                    4 * ngf,
#                                    kernel,
#                                    stride,
#                                    padding,
#                                    padding_mode="reflect",
#                                    bias=bias)

#         stride = 1
#         padding = (kernel - 1) // 2
#         self.upconv3 = nn.Conv2d(4 * ngf,
#                                  2 * ngf,
#                                  kernel,
#                                  stride,
#                                  padding,
#                                  padding_mode="reflect",
#                                  bias=bias)
#         self.upconv2 = nn.Conv2d(2 * ngf,
#                                  ngf,
#                                  kernel,
#                                  stride,
#                                  padding,
#                                  padding_mode="reflect",
#                                  bias=bias)
#         self.upconv1 = nn.Conv2d(ngf,
#                                  nc,
#                                  kernel,
#                                  stride,
#                                  padding,
#                                  padding_mode="reflect",
#                                  bias=bias)

#     def forward(self, x):
#         # downsampling
#         x = self.downconv1(x)
#         x = F.relu(x, inplace=True)
#         x = self.downconv2(x)
#         x = F.relu(x, inplace=True)
#         x = self.downconv3(x)
#         x = F.relu(x, inplace=True)

#         # upsampling
#         x = F.upsample_nearest(x, scale_factor=2)
#         x = self.upconv3(x)
#         x = F.relu(x, inplace=True)
#         x = F.dropout2d(x, p=0.2, inplace=False)
#         x = F.upsample_nearest(x, scale_factor=2)
#         x = self.upconv2(x)
#         x = F.relu(x, inplace=True)
#         x = F.dropout2d(x, p=0.2, inplace=False)
#         x = F.upsample_nearest(x, scale_factor=2)
#         x = self.upconv1(x)
#         x = 0.5 * F.tanh(x)
#         return x


class Generator(nn.Module):
    def __init__(self, nz, ngf=32, nc=3):
        super(Generator, self).__init__()
        bias = True
        kernel = 5
        stride = 1
        padding = (kernel - 1) // 2
        mode = "circular"  # "reflect"

        self.conv1 = nn.Conv2d(nz,
                               ngf,
                               kernel,
                               stride,
                               padding,
                               padding_mode=mode,
                               bias=bias)
        self.conv2 = nn.Conv2d(ngf,
                               ngf,
                               kernel,
                               stride,
                               padding,
                               padding_mode=mode,
                               bias=bias)
        self.conv3 = nn.Conv2d(ngf,
                               ngf,
                               kernel,
                               stride,
                               padding,
                               padding_mode=mode,
                               bias=bias)
        self.conv4 = nn.Conv2d(ngf,
                               ngf,
                               kernel,
                               stride,
                               padding,
                               padding_mode=mode,
                               bias=bias)
        self.conv5 = nn.Conv2d(ngf,
                               3,
                               kernel,
                               stride,
                               padding,
                               padding_mode=mode,
                               bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3(x)
        x = F.relu(x, inplace=True)
        x = self.conv4(x)
        x = F.relu(x, inplace=True)
        x = self.conv5(x)
        x = 0.5 * torch.tanh(x)
        return x


train_dataset = TextureMapDataset(config, "train", {})
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               drop_last=False)

# model = TextureGAN(num_input_channels, 3, config.model.input_texture_ngf)
# model = get_texture_nets()
nz = 4
model = Generator(nz=2 * nz, ngf=64)
model.apply(weights_init)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.lr,
                             betas=(0.9, 0.999))
regression_loss = RegressionLossHelper(config.regression_loss_type)

with torch.no_grad():
    sample = next(iter(train_dataloader))
    train_dataset.apply_batch_transforms(sample)
    h, w = sample["partial_texture"].shape[2:]
    # input_noise = get_noise(32, "noise",
    #                         sample["partial_texture"].shape[2:]).type(
    #                             sample["partial_texture"].type()).detach()
    # input_noise = get_noise(nz, "ones", np.array([4, 4])).type(
    #     sample["partial_texture"].type()).detach()
    input_noise = get_noise(nz, "pe", np.array([128, 128])).type(
        sample["partial_texture"].type()).detach()
    # sinusoidal input?
    # input_noise = torch.ones(1, num_input_channels, h, w)
    # mask = torch.any(
    #     ~sample["partial_texture"].eq(sample["partial_texture"][0:1, :, 0:1, 0:1]),
    #     dim=1,
    #     keepdim=True)

    # This is a hack
    mask = torch.all(sample["partial_texture"] > -0.4, dim=1, keepdim=True)

    print(
        f'sample["partial_texture"] in [{sample["partial_texture"].min():.3f}, {sample["partial_texture"].max():.3f}]'
    )
    print(
        f'sample["texture"] in [{sample["texture"].min():.3f}, {sample["texture"].max():.3f}]'
    )

# Main optimization loop
iterations = 5000
plot_interval = 20
for i in range(1, iterations):
    optimizer.zero_grad()
    pred = model(input_noise)
    # mask = torch.any(
    #     sample["partial_texture"] == sample["partial_texture"][0, :, 0, 0],
    #     dim=1,
    #     keepdim=True)
    loss_regression = regression_loss.calculate_loss(sample["partial_texture"],
                                                     pred *
                                                     mask.detach()).mean()
    loss_regression.backward()
    optimizer.step()

    # visualization
    with torch.no_grad():
        if i % plot_interval == 0:
            print(
                f'loss_regression={loss_regression:.6f}, pred in [{pred.min():.3f}, {pred.max():.3f}]'
            )

            plt.subplot(221)
            input = torch_to_np(sample["partial_texture"].clone())
            input = train_dataset.get_colored_data_for_visualization(input)
            plt.imshow(input)
            plt.axis("off")
            plt.title("partial_texture")
            plt.draw()

            plt.subplot(222)
            _mask = torch_to_np(mask.float().repeat((1, 3, 1, 1)).clone())
            # _mask = train_dataset.get_colored_data_for_visualization(_mask)
            plt.imshow(_mask)
            plt.axis("off")
            plt.title("mask")
            plt.draw()

            plt.subplot(223)
            gt = torch_to_np(sample["texture"].clone())
            gt = train_dataset.get_colored_data_for_visualization(gt)
            plt.imshow(gt)
            plt.axis("off")
            plt.title("gt")
            plt.draw()

            plt.subplot(224)
            pred = torch_to_np(pred)
            pred = train_dataset.get_colored_data_for_visualization(pred)
            plt.imshow(pred)
            plt.axis("off")
            plt.title("prediction")
            plt.draw()

            plt.pause(1e-2)
plt.show()
