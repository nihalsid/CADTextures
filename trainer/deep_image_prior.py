import matplotlib.pyplot as plt
import torch
import numpy as np
from omegaconf.dictconfig import DictConfig

from dataset.texture_map_dataset import TextureMapDataset
from model.texture_gan import TextureGAN
from util.regression_loss import RegressionLossHelper


def torch_to_np(img_torch):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  W x H x C [0..1]
    '''
    img_np = img_torch.detach().cpu().numpy()[0]
    img_np = np.moveaxis(img_np, 0, 2)
    return img_np


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
train_dataset = TextureMapDataset(config, "train", {})
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               drop_last=False)
model = TextureGAN(3, 3, config.model.input_texture_ngf)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.lr,
                             betas=(0.5, 0.999))
regression_loss = RegressionLossHelper(config.regression_loss_type)

sample = next(iter(train_dataloader))
train_dataset.apply_batch_transforms(sample)

print(
    f'sample["partial_texture"] in [{sample["partial_texture"].min():.3f}, {sample["partial_texture"].max()}:.3f]'
)
print(
    f'sample["texture"] in [{sample["texture"].min():.3f}, {sample["texture"].max():.3f}]'
)

# Main optimization loop
iterations = 1000
plot_interval = 10
for i in range(1, iterations):
    optimizer.zero_grad()
    pred = model(sample["partial_texture"])
    mask = sample["mask_texture"]
    loss_regression = regression_loss.calculate_loss(sample["texture"] * mask,
                                                     pred * mask).mean()
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
            # plt.show()
