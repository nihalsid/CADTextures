import matplotlib.pyplot as plt
import torch
import torch.nn as nn
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
        'color_space': 'lab'
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
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
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
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.ones(shape)
    else:
        assert False

    return net_input


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, nz, ngf=64, nc=3):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, 8 * ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(8 * ngf, 4 * ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(4 * ngf, 2 * ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(2 * ngf, ngf, 4, 2, 1, bias=False),
            # nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return 0.5 * self.main(input)


train_dataset = TextureMapDataset(config, "train", {})
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               drop_last=False)

# model = TextureGAN(num_input_channels, 3, config.model.input_texture_ngf)
# model = get_texture_nets()
nz = 1
model = Generator(nz=nz)
model.apply(weights_init)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.lr,
                             betas=(0.5, 0.999))
regression_loss = RegressionLossHelper(config.regression_loss_type)

sample = next(iter(train_dataloader))
train_dataset.apply_batch_transforms(sample)
h, w = sample["partial_texture"].shape[2:]
# input_noise = get_noise(32, "noise",
#                         sample["partial_texture"].shape[2:]).type(
#                             sample["partial_texture"].type()).detach()
input_noise = get_noise(nz, "ones", np.array([4, 4])).type(
    sample["partial_texture"].type()).detach()
# input_noise = torch.ones(1, num_input_channels, h, w)

print(
    f'sample["partial_texture"] in [{sample["partial_texture"].min():.3f}, {sample["partial_texture"].max()}:.3f]'
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
    loss_regression = regression_loss.calculate_loss(
        sample["partial_texture"], pred * sample["mask_texture"]).mean()
    loss_regression.backward()
    optimizer.step()

    # visualization
    with torch.no_grad():
        if i % plot_interval == 0:
            print(
                f'loss_regression={loss_regression:.6f}, pred in [{pred.min():.3f}, {pred.max():.3f}]'
            )

            plt.subplot(131)
            input = torch_to_np(sample["partial_texture"].clone())
            input = train_dataset.get_colored_data_for_visualization(input)
            plt.imshow(input)
            plt.axis("off")
            plt.title("input")
            plt.draw()

            plt.subplot(132)
            gt = torch_to_np(sample["texture"].clone())
            gt = train_dataset.get_colored_data_for_visualization(gt)
            plt.imshow(gt)
            plt.axis("off")
            plt.title("gt")
            plt.draw()

            plt.subplot(133)
            pred = torch_to_np(pred)
            pred = train_dataset.get_colored_data_for_visualization(pred)
            plt.imshow(pred)
            plt.axis("off")
            plt.title("prediction")
            plt.draw()

            plt.pause(1e-2)
plt.show()
