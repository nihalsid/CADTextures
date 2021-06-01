import torch
from omegaconf.dictconfig import DictConfig

from dataset.texture_map_dataset import TextureMapDataset
from model.texture_gan import TextureGAN, get_model
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
train_dataset = TextureMapDataset(config, "train", {})
train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=False,
                                               num_workers=0,
                                               pin_memory=True,
                                               drop_last=False)
sample = next(iter(train_dataloader))
train_dataset.apply_batch_transforms(sample)
model = TextureGAN(3, 3, config.model.input_texture_ngf)
optimizer = torch.optim.Adam(model.parameters(),
                             lr=config.lr,
                             betas=(0.5, 0.999))
regression_loss = RegressionLossHelper(config.regression_loss_type)

iterations = 10
for i in range(iterations):
    optimizer.zero_grad()
    pred = model(sample["partial_texture"])
    loss_regression = regression_loss.calculate_loss(sample["texture"],
                                                     pred).mean()
    loss_regression.backward()
    print(loss_regression)
    optimizer.step()
