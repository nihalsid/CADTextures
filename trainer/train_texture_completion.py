from pathlib import Path
from random import randint
import hydra
import pytorch_lightning as pl
import torch
import os
import numpy as np

from dataset.texture_completion_dataset import TextureCompletionDataset
from dataset.texture_map_dataset import TextureMapDataset
from model.discriminator import get_discriminator, get_discriminator_local
from model.texture_gan import TextureGAN
from trainer.train_texture_map_predictor import TextureMapPredictorModule
from util.feature_loss import FeatureLossHelper
from util.gan_loss import GANLoss
from util.regression_loss import RegressionLossHelper


class TextureCompletionModule(pl.LightningModule):

    def __init__(self, config):
        super(TextureCompletionModule, self).__init__()
        self.hparams = config
        dataset = lambda split: TextureCompletionDataset(config, split)
        self.train_dataset, self.val_dataset, self.train_val_dataset, self.train_vis_dataset, self.val_vis_dataset = dataset('train'), dataset('val'), dataset('train_val'), dataset('train_vis'), dataset('val_vis')
        self.model = TextureGAN(4, 3, config.model.input_texture_ngf)
        self.discriminator = get_discriminator(config)
        self.discriminator_local = get_discriminator_local(config)
        self.gan_loss = GANLoss(self.hparams.gan_loss_type)
        self.regression_loss = RegressionLossHelper(self.hparams.regression_loss_type)
        self.feature_loss_helper = FeatureLossHelper(['relu4_2'], ['relu3_2', 'relu4_2'])

    def forward(self, batch):
        self.train_dataset.apply_batch_transforms(batch)
        input_maps = torch.cat([batch['input'], batch['mask']], dim=1)
        generated_texture = self.model(input_maps)
        return TextureMapDataset.apply_mask_texture(generated_texture, batch['mask'])

    def calculate_losses(self, generated_texture, batch):
        gt_texture_l, gt_texture_ab = TextureMapPredictorModule.split_into_channels(batch['target'])
        generated_texture_l, generated_texture_ab = TextureMapPredictorModule.split_into_channels(generated_texture)
        loss_regression_l = self.regression_loss.calculate_loss(gt_texture_l, generated_texture_l).mean()
        loss_regression_ab = self.regression_loss.calculate_loss(gt_texture_ab, generated_texture_ab).mean()
        loss_content = self.feature_loss_helper.calculate_feature_loss(gt_texture_l, generated_texture_l).mean()
        style_loss_maps = self.feature_loss_helper.calculate_style_loss(gt_texture_l, generated_texture_l)
        loss_style = style_loss_maps[0].mean() + style_loss_maps[1].mean()
        loss_gan = self.gan_loss.compute_generator_loss(self.discriminator, generated_texture_l)
        patch_gt_l, patch_generated_l = TextureMapDataset.sample_patches(batch['mask'], self.hparams.patch_size, self.hparams.num_patches, gt_texture_l, generated_texture_l)
        loss_gan_local = self.gan_loss.compute_generator_loss(self.discriminator_local, torch.cat([patch_gt_l, patch_generated_l], dim=1))
        loss_total = loss_regression_l * self.hparams.lambda_regr_l + loss_regression_ab * self.hparams.lambda_regr_ab + loss_content * self.hparams.lambda_content + loss_style * self.hparams.lambda_style \
                     + loss_gan * self.hparams.lambda_g + loss_gan_local * self.hparams.lambda_g_local
        return loss_total, loss_regression_l, loss_regression_ab, loss_content, loss_style, loss_gan, loss_gan_local

    def training_step(self, batch, batch_index, optimizer_idx):
        if optimizer_idx == 0:
            generated_texture = self.forward(batch)
            loss_total, loss_regression_l, loss_regression_ab, loss_content, loss_style, loss_gan, loss_gan_local = self.calculate_losses(generated_texture, batch)
            self.log('train/loss_regression_l', loss_regression_l, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log('train/loss_regression_ab', loss_regression_ab, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log('train/loss_content', loss_content, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log('train/loss_style', loss_style, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log('train/loss_total', loss_total, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'adversarial/G', loss_gan, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'adversarial_local/G', loss_gan_local, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        else:
            loss_total = torch.zeros([1], dtype=batch['target'].dtype, requires_grad=True)
            if int(self.hparams.lambda_g > 0):
                with torch.no_grad():
                    generated_texture = self.forward(batch)
                gt_texture_l, _ = TextureMapPredictorModule.split_into_channels(batch['target'])
                generated_texture_l, _ = TextureMapPredictorModule.split_into_channels(generated_texture)
                if optimizer_idx == 1:
                    d_real_loss, d_fake_loss, d_gp_loss = self.gan_loss.compute_discriminator_loss(self.discriminator, gt_texture_l, generated_texture_l.detach())
                else:
                    patch_gt_l_0, patch_gt_l_1, patch_generated_l = TextureMapDataset.sample_patches(batch['mask'], self.hparams.patch_size, self.hparams.num_patches, gt_texture_l, gt_texture_l, generated_texture_l)
                    d_real_loss, d_fake_loss, d_gp_loss = self.gan_loss.compute_discriminator_loss(self.discriminator_local, torch.cat([patch_gt_l_0, patch_gt_l_1], dim=1), torch.cat([patch_gt_l_0, patch_generated_l], dim=1).detach())
                loss_total = d_real_loss + d_fake_loss + self.hparams.lambda_gp * d_gp_loss
                suffix = ["", "_local"][optimizer_idx - 1]
                self.log(f'adversarial{suffix}/D_real', d_real_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
                self.log(f'adversarial{suffix}/D_fake', d_fake_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
                self.log(f'adversarial{suffix}/D_gp', d_gp_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
                self.log(f'adversarial{suffix}/D', d_real_loss + d_fake_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        return {'loss': loss_total}

    def validation_step(self, batch, batch_index, dataloader_index):
        split = ["val", "train"][dataloader_index]
        suffix = ["", "_epoch"][dataloader_index]
        generated_texture = self.forward(batch)
        loss_total, loss_regression_l, loss_regression_ab, loss_content, loss_style, _, _ = self.calculate_losses(generated_texture, batch)
        self.log(f'{split}/loss_regression_l{suffix}', loss_regression_l, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f'{split}/loss_regression_ab{suffix}', loss_regression_ab, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f'{split}/loss_content{suffix}', loss_content, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f'{split}/loss_style{suffix}', loss_style, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f'{split}/loss_total{suffix}', loss_total, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

    def validation_epoch_end(self, _outputs):
        if int(os.environ.get('LOCAL_RANK', 0)) == 0:
            visualization_datasets = [self.val_vis_dataset, self.train_vis_dataset]
            dataset_names = ['val', 'train']
            for ds_idx, ds in enumerate(visualization_datasets):
                output_vis_path = Path("runs") / self.hparams['experiment'] / f"vis_{dataset_names[ds_idx]}" / f'{(self.global_step // 1000):05d}'
                (output_vis_path / "figures").mkdir(exist_ok=True, parents=True)
                (output_vis_path / "meshes").mkdir(exist_ok=True, parents=True)
                loader = torch.utils.data.DataLoader(ds, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)
                for batch_idx, batch in enumerate(loader):
                    TextureCompletionDataset.move_batch_to_gpu(batch, self.device)
                    generated_texture = self.forward(batch)
                    gt_texture_l, gt_texture_ab = TextureMapPredictorModule.split_into_channels(batch['target'])
                    generated_texture_l, generated_texture_ab = TextureMapPredictorModule.split_into_channels(generated_texture)
                    loss_regression_l = self.regression_loss.calculate_loss(gt_texture_l, generated_texture_l).mean(axis=1).squeeze(1)
                    loss_regression_ab = self.regression_loss.calculate_loss(gt_texture_ab, generated_texture_ab).mean(axis=1).squeeze(1)
                    loss_content = self.feature_loss_helper.calculate_feature_loss(gt_texture_l, generated_texture_l).mean(axis=1).squeeze(1)
                    style_loss_maps = self.feature_loss_helper.calculate_style_loss(gt_texture_l, generated_texture_l)
                    loss_style = (torch.nn.functional.interpolate(style_loss_maps[1].unsqueeze(1), size=style_loss_maps[0].shape[1:]).squeeze(1) + style_loss_maps[0]) / 2
                    for ii in range(generated_texture.shape[0]):
                        self.visualize_prediction(output_vis_path, batch['name'][ii], batch['view_index'][ii], batch['input'][ii].cpu().numpy(), batch['target'][ii].cpu().numpy(),
                                                  generated_texture[ii].cpu().numpy(), loss_regression_l[ii].cpu().numpy(), loss_regression_ab[ii].cpu().numpy(), loss_content[ii].cpu().numpy(), loss_style[ii].cpu().numpy())

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d_local = torch.optim.Adam(self.discriminator_local.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d, opt_d_local], []

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return [torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=False),
                torch.utils.data.DataLoader(self.train_val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=False)]

    def visualize_prediction(self, save_dir, name, v_idx, incomplete, target, prediction, loss_regression_l, loss_regression_ab, loss_content, loss_style):
        import matplotlib.pyplot as plt
        incomplete = self.train_dataset.denormalize_and_rgb(np.transpose(incomplete, (1, 2, 0)))
        target = self.train_dataset.denormalize_and_rgb(np.transpose(target, (1, 2, 0)))
        prediction = self.train_dataset.denormalize_and_rgb(np.transpose(prediction, (1, 2, 0)))
        f, axarr = plt.subplots(1, 7, figsize=(28, 4))
        items = [incomplete, target, prediction]
        for i in range(3):
            axarr[i].imshow(items[i])
            axarr[i].axis('off')
        items = [loss_regression_l, loss_regression_ab, loss_content, loss_style]
        for i in range(4):
            items[i] = (items[i] - items[i].min()) / (items[i].max() - items[i].min())
            axarr[3 + i].imshow(1 - items[i], cmap='RdYlGn')
            axarr[3 + i].axis('off')
        plt.savefig(save_dir / "figures" / f"{name}_{v_idx}.jpg", bbox_inches='tight', dpi=240)
        plt.close()

    def on_post_move_to_device(self):
        self.feature_loss_helper.move_to_device(self.device)


@hydra.main(config_path='../config', config_name='completion')
def main(config):
    from datetime import datetime
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    from util.filesystem_logger import FilesystemLogger
    ds_name = '_'.join(config.dataset.name.split('/'))
    config.experiment = f"{datetime.now().strftime('%d%m%H%M')}_{ds_name}_{config['experiment']}"
    if not config.wandb_main and config.suffix == '':
        config.suffix = '-dev'
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)

    seed_everything(config.seed)
    # noinspection PyUnusedLocal
    filesystem_logger = FilesystemLogger(config)
    logger = WandbLogger(project=f'TextureCompletion{config.suffix}[{ds_name}]', name=config.experiment, id=config.experiment)

    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / config.experiment), filename='_ckpt_{epoch}', save_top_k=-1, verbose=False, period=config.save_epoch)
    model = TextureCompletionModule(config)
    trainer = Trainer(gpus=[0], num_sanity_val_steps=config.sanity_steps, max_epochs=config.max_epoch, limit_val_batches=config.val_check_percent, callbacks=[checkpoint_callback],
                      val_check_interval=float(min(config.val_check_interval, 1)), check_val_every_n_epoch=max(1, config.val_check_interval), resume_from_checkpoint=config.resume, logger=logger, benchmark=True)
    trainer.fit(model)


if __name__ == '__main__':
    main()
