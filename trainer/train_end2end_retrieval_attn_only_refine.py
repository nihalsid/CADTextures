from random import randint

import hydra
import pytorch_lightning as pl
import torch
from pathlib import Path
import wandb

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only

from dataset.texture_end2end_dataset import TextureEnd2EndDataset
from dataset.texture_map_dataset import TextureMapDataset
from model.fold import Fold2D
from model.diffusion_model import Decoder, Encoder
from trainer.train_texture_map_predictor import TextureMapPredictorModule
from util.feature_loss import FeatureLossHelper
from util.misc import cosine_decay
from util.regression_loss import RegressionLossHelper


class TextureEnd2EndModule(pl.LightningModule):

    def __init__(self, config):
        super(TextureEnd2EndModule, self).__init__()
        self.save_hyperparameters(config)
        self.preload_dict = {}
        assert config.dataset.texture_map_size == 128, "only 128x128 texture map supported"
        encoder = lambda in_channels, z_channels: Encoder(ch=128, out_ch=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, attn_resolutions=[8], dropout=0.0, resamp_with_conv=True, in_channels=in_channels, resolution=128, z_channels=z_channels, double_z=False)
        self.fenc_input, self.fenc_target = encoder(4, config.fenc_zdim), encoder(3, config.fenc_zdim)
        self.regression_loss = RegressionLossHelper(self.hparams.regression_loss_type)
        self.feature_loss_helper = FeatureLossHelper(['relu4_2'], ['relu3_2', 'relu4_2'])
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.dataset = lambda split: TextureEnd2EndDataset(config, split, self.preload_dict)
        self.train_dataset = self.dataset('train')
        self.decoder = Decoder(ch=128, out_ch=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, attn_resolutions=[8], dropout=0.0, resamp_with_conv=True, in_channels=3, resolution=128, z_channels=config.fenc_zdim, double_z=False)
        self.attention_layers = torch.nn.ModuleList()
        for _att_i in range(5):
            self.attention_layers.append(torch.nn.MultiheadAttention(embed_dim=256, num_heads=1, batch_first=True)) 
        self.fold = Fold2D(config.dataset.texture_map_size // config.dictionary.patch_size, config.dictionary.patch_size, 3)
        self.fold_features = Fold2D(config.dataset.texture_map_size // config.dictionary.patch_size, 1, config.fenc_zdim)
        self.fold_s = Fold2D(config.dataset.texture_map_size // config.dictionary.patch_size, 1, config.dictionary.K)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.fenc_input.parameters()) + list(self.fenc_target.parameters()) + list(self.attention_layers.parameters()) + list(self.decoder.parameters()), lr=self.hparams.lr, betas=(0.5, 0.9))
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
        features_in = self.fenc_input(torch.cat([batch['partial_texture'], batch['mask_missing']], 1))
        features_candidates = self.fenc_target(batch['database_textures'])
        refinement, s, b = self.attention_blending_decode(batch['mask_texture'], features_in, features_candidates.unsqueeze(1))
        return refinement, s, b

    def attention_blending_decode(self, mask_texture, features_in, knn_candidate_features, return_debug_vis=False):
        cumulative_output = features_in.unsqueeze(1)
        cumulative_weights = torch.zeros((features_in.shape[0], 1, self.hparams.dictionary.K)).to(features_in.device)
        for att_i in range(5):
            attn_output, attn_weights = self.attention_layers[att_i](cumulative_output, knn_candidate_features, knn_candidate_features)
            cumulative_weights = cumulative_weights + attn_weights
            cumulative_output = cumulative_output + attn_output
        cumulative_weights = cumulative_weights / 5
        cumulative_output, cumulative_weights = cumulative_output.squeeze(1), cumulative_weights.squeeze(1)
        
        o = self.fold_features(cumulative_output.view(cumulative_output.shape[0], cumulative_output.shape[1], 1, 1))
        refinement = TextureMapDataset.apply_mask_texture(self.decoder(o), mask_texture)
        if not return_debug_vis:
            return refinement, self.fold_s(attn_weights.unsqueeze(-1).unsqueeze(-1)), self.fold_s(attn_weights.unsqueeze(-1).unsqueeze(-1))
        else:
            o_0 = self.fold_features(features_in.view(cumulative_output.shape[0], cumulative_output.shape[1], 1, 1))
            o_1 = o - o_0
            refinement = TextureMapDataset.apply_mask_texture(self.decoder(o), mask_texture)
            refinement_noret = TextureMapDataset.apply_mask_texture(self.decoder(o_0), mask_texture)
            refinement_noinp = TextureMapDataset.apply_mask_texture(self.decoder(o_1), mask_texture)
            return refinement, refinement_noret, refinement_noinp, self.fold_s(attn_weights.unsqueeze(-1).unsqueeze(-1)), self.fold_s(attn_weights.unsqueeze(-1).unsqueeze(-1))

    def training_step(self, batch, batch_idx):
        self.train_dataset.apply_batch_transforms(batch)
        gt_texture_l, gt_texture_ab = TextureMapPredictorModule.split_into_channels(batch['texture'])
        refinement, score, blend = self.forward(batch)
        refined_texture_l, refined_texture_ab = TextureMapPredictorModule.split_into_channels(refinement)
        loss_regression_ref_l = self.regression_loss.calculate_loss(gt_texture_l, refined_texture_l).mean()
        loss_regression_ref_ab = self.regression_loss.calculate_loss(gt_texture_ab, refined_texture_ab).mean()
        loss_content_ref = self.feature_loss_helper.calculate_feature_loss(gt_texture_l, refined_texture_l).mean()
        style_loss_maps = self.feature_loss_helper.calculate_style_loss(gt_texture_l, refined_texture_l)
        loss_style_ref = style_loss_maps[0].mean() + style_loss_maps[1].mean()
        # print(f"{loss_regression_ref_l.item()},{loss_regression_ref_ab.item()},{loss_content_ref.item()},{loss_style_ref.item()}")
        self.log("train/loss_regression_ref_l", loss_regression_ref_l * self.hparams.lambda_regr_l, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/loss_regression_ref_ab", loss_regression_ref_ab * self.hparams.lambda_regr_ab, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/loss_style_ref", loss_style_ref * self.hparams.lambda_style, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/loss_content_ref", loss_content_ref * self.hparams.lambda_content, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        loss_total = loss_regression_ref_l * cosine_decay(self.current_epoch, 20, self.hparams.lambda_regr_l, 2) + loss_regression_ref_ab * cosine_decay(self.current_epoch, 20, self.hparams.lambda_regr_ab, 10) + loss_content_ref * self.hparams.lambda_content + loss_style_ref * self.hparams.lambda_style
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

                features_in = self.fenc_input(torch.cat([batch['partial_texture'], batch['mask_missing']], 1))
                features_candidates = self.fenc_target(batch['database_textures'])

                refinement, refinement_noret, refinement_noinp, s, b = self.attention_blending_decode(batch['mask_texture'], features_in.to(self.device), features_candidates.unsqueeze(1).to(self.device), return_debug_vis=True)

                ds_vis.visualize_texture_batch_01(batch['partial_texture'].cpu().numpy(), batch['texture'].cpu().numpy(),  self.fold(batch['database_textures']).unsqueeze(1).cpu().numpy(), refinement_noret.cpu().numpy(), refinement_noinp.cpu().numpy(), refinement.cpu().numpy(),
                                                  s.cpu().numpy(), s.cpu().numpy(), lambda prefix: output_dir / "val_vis" / f"{prefix}_{batch_idx:04d}.jpg")
                total_loss_ref_regression += self.mse_loss(refinement.to(self.device), batch['texture']).cpu().item()

        total_loss_ref_regression /= len(ds_vis)
        self.log("val/loss_ref_regression", total_loss_ref_regression, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True)

    def on_train_start(self):
        self.feature_loss_helper.move_to_device(self.device)

    def on_validation_start(self):
        self.feature_loss_helper.move_to_device(self.device)


@hydra.main(config_path='../config', config_name='texture_end2end_attn_only_refine')
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
    logger = WandbLogger(project=f'End2End{config.suffix}[{ds_name}]', name=config.experiment, id=config.experiment, settings=wandb.Settings(start_method='thread'))
    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / config.experiment), filename='_ckpt_{epoch}', save_top_k=-1, verbose=False, every_n_epochs=config.save_epoch)
    model = TextureEnd2EndModule(config)
    trainer = Trainer(gpus=-1, accelerator='ddp', plugins=DDPPlugin(find_unused_parameters=True), num_sanity_val_steps=config.sanity_steps, max_epochs=config.max_epoch, limit_val_batches=config.val_check_percent,
                      callbacks=[checkpoint_callback],
                      val_check_interval=float(min(config.val_check_interval, 1)), check_val_every_n_epoch=max(1, config.val_check_interval), resume_from_checkpoint=config.resume, logger=logger, benchmark=True)
    trainer.fit(model)


if __name__ == '__main__':
    main()
