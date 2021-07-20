from random import randint

import hydra
import pytorch_lightning as pl
import torch
from pathlib import Path
import numpy as np

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only

from dataset.texture_end2end_dataset import TextureEnd2EndDataset
from dataset.texture_map_dataset import TextureMapDataset
from model.diffusion_model import Decoder, DeformableEncoder
from trainer.train_texture_map_predictor import TextureMapPredictorModule
from util.feature_loss import FeatureLossHelper
from util.regression_loss import RegressionLossHelper


class TextureRegressionModule(pl.LightningModule):

    def __init__(self, config):
        super(TextureRegressionModule, self).__init__()
        self.save_hyperparameters(config)
        self.preload_dict = {}
        assert config.dataset.texture_map_size == 128, "only 128x128 texture map supported"
        encoder = lambda in_channels, z_channels: DeformableEncoder(ch=128, out_ch=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, attn_resolutions=[8], dropout=0.0, resamp_with_conv=True, in_channels=in_channels, resolution=128, z_channels=z_channels, double_z=False)
        self.fenc_input = encoder(4, config.fenc_zdim)
        self.regression_loss = RegressionLossHelper(self.hparams.regression_loss_type)
        self.feature_loss_helper = FeatureLossHelper(['relu4_2'], ['relu3_2', 'relu4_2'])
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.dataset = lambda split: TextureEnd2EndDataset(config, split, self.preload_dict)
        self.train_dataset = self.dataset('train')
        self.decoder = Decoder(ch=128, out_ch=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, attn_resolutions=[8], dropout=0.0, resamp_with_conv=True, in_channels=3, resolution=128, z_channels=config.fenc_zdim, double_z=False)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.fenc_input.parameters()) + list(self.decoder.parameters()), lr=self.hparams.lr, betas=(0.5, 0.9))
        scheduler = []
        if self.hparams.scheduler is not None:
            scheduler = [torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.scheduler, gamma=0.5)]
        return [optimizer], scheduler

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        dataset = self.dataset('train')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False, pin_memory=True)

    def forward(self, batch):
        features_in, offsets = self.fenc_input(torch.cat([batch['partial_texture'], batch['mask_missing']], 1))
        refinement = TextureMapDataset.apply_mask_texture(self.decoder(features_in), batch['mask_texture'])
        return refinement, offsets

    def training_step(self, batch, batch_idx):
        self.train_dataset.apply_batch_transforms(batch)
        gt_texture_l, gt_texture_ab = TextureMapPredictorModule.split_into_channels(batch['texture'])
        loss_total = torch.zeros([1, ], device=self.device)
        refinement, _ = self.forward(batch)
        refined_texture_l, refined_texture_ab = TextureMapPredictorModule.split_into_channels(refinement)
        loss_regression_ref_l = self.regression_loss.calculate_loss(gt_texture_l, refined_texture_l).mean()
        loss_regression_ref_ab = self.regression_loss.calculate_loss(gt_texture_ab, refined_texture_ab).mean()
        loss_content_ref = self.feature_loss_helper.calculate_feature_loss(gt_texture_l, refined_texture_l).mean()
        style_loss_maps = self.feature_loss_helper.calculate_style_loss(gt_texture_l, refined_texture_l)
        loss_style_ref = style_loss_maps[0].mean() + style_loss_maps[1].mean()
        self.log("train/loss_regression_ref_l", loss_regression_ref_l * self.hparams.lambda_regr_l, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/loss_regression_ref_ab", loss_regression_ref_ab * self.hparams.lambda_regr_ab, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/loss_style_ref", loss_style_ref * self.hparams.lambda_style, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/loss_content_ref", loss_content_ref * self.hparams.lambda_content, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        loss_total = loss_total + (loss_regression_ref_l * self.hparams.lambda_regr_l + loss_regression_ref_ab * self.hparams.lambda_regr_ab + loss_content_ref * self.hparams.lambda_content + loss_style_ref * self.hparams.lambda_style / 25)
        return loss_total

    def validation_step(self, batch, batch_idx):
        pass

    @rank_zero_only
    def validation_epoch_end(self, outputs):
        dataset = lambda split: TextureEnd2EndDataset(self.hparams, split, self.preload_dict, single_view=True)
        output_dir = Path("runs") / self.hparams.experiment / "visualization" / f"epoch_{self.current_epoch:04d}"
        output_dir.mkdir(exist_ok=True, parents=True)
        (output_dir / "val_vis").mkdir(exist_ok=True)

        ds_vis = dataset('val_vis')
        loader = torch.utils.data.DataLoader(ds_vis, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False, pin_memory=True)
        total_loss_ref_regression = 0

        with torch.no_grad():

            for batch_idx, batch in enumerate(loader):
                ds_vis.move_batch_to_gpu(batch, self.device)
                ds_vis.apply_batch_transforms(batch)
                refinement, offsets = self.forward(batch)
                ds_vis.visualize_texture_batch_02(batch['partial_texture'].cpu().numpy(), batch['texture'].cpu().numpy(), refinement.cpu().numpy(), offsets.cpu().numpy(), lambda prefix: output_dir / "val_vis" / f"{prefix}_{batch_idx:04d}.jpg")
                total_loss_ref_regression += self.mse_loss(refinement.to(self.device), batch['texture']).cpu().item()

        total_loss_ref_regression /= len(ds_vis)
        self.log("val/loss_ref_regression", total_loss_ref_regression, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def on_train_start(self):
        self.feature_loss_helper.move_to_device(self.device)


@hydra.main(config_path='../config', config_name='texture_deform')
def main(config):
    from datetime import datetime
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    from util.filesystem_logger import FilesystemLogger
    ds_name = '_'.join(config.dataset.name.split('/'))
    if not config.wandb_main and config.suffix == '':
        config.suffix = '-dev'
    config.experiment = f"{datetime.now().strftime('%d%m%H%M')}_end2end_{config['experiment']}"
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)

    seed_everything(config.seed)
    # noinspection PyUnusedLocal
    filesystem_logger = FilesystemLogger(config)
    logger = WandbLogger(project=f'End2End{config.suffix}[{ds_name}]', name=config.experiment, id=config.experiment)
    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / config.experiment), filename='_ckpt_{epoch}', save_top_k=-1, verbose=False, every_n_val_epochs=config.save_epoch)
    model = TextureRegressionModule(config)
    trainer = Trainer(gpus=-1, accelerator='ddp', plugins=DDPPlugin(find_unused_parameters=True), num_sanity_val_steps=config.sanity_steps, max_epochs=config.max_epoch, limit_val_batches=config.val_check_percent,
                      callbacks=[checkpoint_callback],
                      val_check_interval=float(min(config.val_check_interval, 1)), check_val_every_n_epoch=max(1, config.val_check_interval), resume_from_checkpoint=config.resume, logger=logger, benchmark=True)
    trainer.fit(model)


if __name__ == '__main__':
    main()
