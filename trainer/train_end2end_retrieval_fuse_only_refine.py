from random import randint

import hydra
import pytorch_lightning as pl
import torch
from pathlib import Path
import wandb
from pytorch_lightning import Callback

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only

from dataset.texture_end2end_dataset import TextureEnd2EndDataset
from dataset.texture_map_dataset import TextureMapDataset
from model.fold import Fold2D
from model.diffusion_model import Decoder, Encoder
from trainer.train_texture_map_predictor import TextureMapPredictorModule
from util.contrastive_loss import NTXentLoss
from util.feature_loss import FeatureLossHelper
from util.regression_loss import RegressionLossHelper


class TextureEnd2EndModule(pl.LightningModule):

    def __init__(self, config):
        super(TextureEnd2EndModule, self).__init__()
        self.save_hyperparameters(config)
        self.preload_dict = {}
        assert config.dataset.texture_map_size == 128, "only 128x128 texture map supported"
        encoder = lambda in_channels, z_channels: Encoder(ch=128, out_ch=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, attn_resolutions=[8], dropout=0.0, resamp_with_conv=True, in_channels=in_channels, resolution=128, z_channels=z_channels,
                                                          double_z=False)
        self.fenc_input, self.fenc_target = encoder(4, config.fenc_zdim), encoder(3, config.fenc_zdim)
        self.nt_xent_loss = NTXentLoss(float(config.temperature), config.dataset.texture_map_size // config.dictionary.patch_size, True)
        self.regression_loss = RegressionLossHelper(self.hparams.regression_loss_type)
        self.feature_loss_helper = FeatureLossHelper(['relu4_2'], ['relu3_2', 'relu4_2'], 'lab')
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.dataset = lambda split: TextureEnd2EndDataset(config, split, self.preload_dict)
        self.train_dataset = self.dataset('train')
        self.decoder = Decoder(ch=128, out_ch=3, ch_mult=(1, 1, 2, 2, 4), num_res_blocks=2, attn_resolutions=[8], dropout=0.0, resamp_with_conv=True, in_channels=3, resolution=128, z_channels=config.fenc_zdim, double_z=False)

        self.init_scale = 14
        self.init_shift = -5
        self.sig_scale = torch.nn.Parameter(torch.ones(1) * self.init_scale)
        self.sig_shift = torch.nn.Parameter(torch.ones(1) * self.init_shift)
        self.sigmoid = torch.nn.Sigmoid()

        self.fold = Fold2D(config.dataset.texture_map_size // config.dictionary.patch_size, config.dictionary.patch_size, 3)
        self.fold_features = Fold2D(config.dataset.texture_map_size // config.dictionary.patch_size, 1, config.fenc_zdim)
        self.fold_s = Fold2D(config.dataset.texture_map_size // config.dictionary.patch_size, 1, config.dictionary.K)
        self.fold_b = Fold2D(config.dataset.texture_map_size // config.dictionary.patch_size, 1, 1)

        self.current_phase = config.current_phase

    def optimizer_input(self):
        optimizer = torch.optim.Adam(list(self.fenc_input.parameters()) + list(self.decoder.parameters()), lr=self.hparams.lr, betas=(0.5, 0.9))
        return [optimizer], []

    def optimizer_target(self):
        optimizer = torch.optim.Adam(list(self.fenc_target.parameters()), lr=self.hparams.lr, betas=(0.5, 0.9))
        return [optimizer], []

    def optimizer_all(self):
        optimizer = torch.optim.Adam(list(self.fenc_input.parameters()) + list(self.decoder.parameters()) + [self.sig_scale, self.sig_shift] + list(self.fenc_target.parameters()), lr=self.hparams.lr, betas=(0.5, 0.9))
        return [optimizer], []

    def configure_optimizers(self):
        return self.get_current_optimizer()()

    def get_current_optimizer(self):
        return [self.optimizer_input, self.optimizer_target, self.optimizer_all][self.current_phase]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        dataset = self.dataset('train')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False, pin_memory=True)

    def forward_in(self, batch):
        features_in = self.fenc_input(torch.cat([batch['partial_texture'], batch['mask_missing']], 1))
        return TextureMapDataset.apply_mask_texture(self.decoder(self.fold_features(features_in.unsqueeze(-1).unsqueeze(-1))), batch['mask_texture']), torch.nn.functional.normalize(features_in, dim=1)

    def forward_tgt(self, batch):
        features_tgt = self.fenc_target(self.train_dataset.unfold(batch['texture']))
        return TextureMapDataset.apply_mask_texture(self.decoder(self.fold_features(features_tgt.unsqueeze(-1).unsqueeze(-1))), batch['mask_texture']), torch.nn.functional.normalize(features_tgt, dim=1)

    def forward_all(self, batch):
        features_in = self.fenc_input(torch.cat([batch['partial_texture'], batch['mask_missing']], 1))
        features_target = self.fenc_target(self.train_dataset.unfold(batch['texture']))
        reconstruction_tgt = TextureMapDataset.apply_mask_texture(self.decoder(self.fold_features(features_target.unsqueeze(-1).unsqueeze(-1))), batch['mask_texture'])
        features_candidates = self.fenc_target(batch['database_textures'])
        refinement, s, b = self.attention_blending_decode(batch['mask_texture'], features_in, features_candidates.unsqueeze(1))
        return refinement, reconstruction_tgt, torch.nn.functional.normalize(features_in, dim=1), torch.nn.functional.normalize(features_target, dim=1)

    def attention_blending_decode(self, mask_texture, features_in, knn_candidate_features, return_debug_vis=False):
        s = torch.einsum('ik,ijk->ij', torch.nn.functional.normalize(features_in, dim=1), torch.nn.functional.normalize(knn_candidate_features, dim=2))
        b = self.sigmoid(s.unsqueeze(1).view(s.shape[0], 1) * self.sig_scale + self.sig_shift)
        o = self.fold_features((knn_candidate_features.squeeze(1) * b + features_in * (1 - b)).view(features_in.shape[0], features_in.shape[1], 1, 1))
        refinement = TextureMapDataset.apply_mask_texture(self.decoder(o), mask_texture)
        if return_debug_vis:
            o_0 = self.fold_features(features_in.view(features_in.shape[0], features_in.shape[1], 1, 1))
            o_1 = self.fold_features(knn_candidate_features.squeeze(1).view(features_in.shape[0], features_in.shape[1], 1, 1))
            refinement_noret = TextureMapDataset.apply_mask_texture(self.decoder(o_0), mask_texture)
            refinement_noinp = TextureMapDataset.apply_mask_texture(self.decoder(o_1), mask_texture)
            return refinement, refinement_noret, refinement_noinp, self.fold_s(s.unsqueeze(-1).unsqueeze(-1)), self.fold_b(b.unsqueeze(-1).unsqueeze(-1))
        return refinement, self.fold_s(s.unsqueeze(-1).unsqueeze(-1)), self.fold_b(b.unsqueeze(-1).unsqueeze(-1))

    def training_step(self, batch, batch_idx):
        self.train_dataset.apply_batch_transforms(batch)
        gt_texture_l, gt_texture_ab = TextureMapPredictorModule.split_into_channels(batch['texture'])
        if self.current_phase == 0:
            refinement, _ = self.forward_in(batch)
            refined_texture_l, refined_texture_ab = TextureMapPredictorModule.split_into_channels(refinement)
            loss_regression_ref_l = self.regression_loss.calculate_loss(gt_texture_l, refined_texture_l).mean()
            loss_regression_ref_ab = self.regression_loss.calculate_loss(gt_texture_ab, refined_texture_ab).mean()
            loss_total = loss_regression_ref_l * self.hparams.lambda_regr_l + \
                         loss_regression_ref_ab * self.hparams.lambda_regr_ab
            self.log("train/loss_regression_ref_l", loss_regression_ref_l * self.hparams.lambda_regr_l, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("train/loss_regression_ref_ab", loss_regression_ref_ab * self.hparams.lambda_regr_ab, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        elif self.current_phase == 1:
            self.fenc_input.eval()
            self.decoder.eval()
            refinement, features_tgt = self.forward_tgt(batch)
            features_in = torch.nn.functional.normalize(self.fenc_input(torch.cat([batch['partial_texture'], batch['mask_missing']], 1)), dim=1).detach()
            refined_texture_l, refined_texture_ab = TextureMapPredictorModule.split_into_channels(refinement)
            loss_regression_ref_l = self.regression_loss.calculate_loss(gt_texture_l, refined_texture_l).mean()
            loss_regression_ref_ab = self.regression_loss.calculate_loss(gt_texture_ab, refined_texture_ab).mean()
            loss_ntxent = self.nt_xent_loss(features_in, features_tgt)
            loss_total = loss_regression_ref_l * self.hparams.lambda_regr_l + \
                         loss_regression_ref_ab * self.hparams.lambda_regr_ab + \
                         loss_ntxent * self.hparams.start_contrastive_weight
            self.log("train/loss_regression_ref_l_tgt", loss_regression_ref_l * self.hparams.lambda_regr_l, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("train/loss_regression_ref_ab_tgt", loss_regression_ref_ab * self.hparams.lambda_regr_ab, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("train/loss_contrastive", loss_ntxent * self.hparams.start_contrastive_weight, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.fenc_input.train()
            self.decoder.train()
        elif self.current_phase == 2:
            refinement, reconstruction_tgt, features_in, features_tgt = self.forward_all(batch)
            refined_texture_l, refined_texture_ab = TextureMapPredictorModule.split_into_channels(refinement)
            reconstructed_texture_l, reconstructed_texture_ab = TextureMapPredictorModule.split_into_channels(reconstruction_tgt)
            loss_ntxent = self.nt_xent_loss(features_in, features_tgt)
            loss_regression_ref_l = self.regression_loss.calculate_loss(gt_texture_l, refined_texture_l).mean()
            loss_regression_ref_ab = self.regression_loss.calculate_loss(gt_texture_ab, refined_texture_ab).mean()
            loss_content_ref = self.feature_loss_helper.calculate_feature_loss(batch['texture'], refinement).mean()
            style_loss_maps = self.feature_loss_helper.calculate_style_loss(batch['texture'], refinement)
            loss_style_ref = style_loss_maps[0].mean() + style_loss_maps[1].mean()
            loss_recontruction_ref_l = self.regression_loss.calculate_loss(gt_texture_l, reconstructed_texture_l).mean()
            loss_recontruction_ref_ab = self.regression_loss.calculate_loss(gt_texture_ab, reconstructed_texture_ab).mean()
            # print(f"{loss_regression_ref_l.item()},{loss_regression_ref_ab.item()},{loss_content_ref.item()},{loss_style_ref.item()}")
            self.log("train/loss_regression_ref_l", loss_regression_ref_l * self.hparams.lambda_regr_l, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("train/loss_regression_ref_ab", loss_regression_ref_ab * self.hparams.lambda_regr_ab, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("train/loss_style_ref", loss_style_ref * self.hparams.lambda_style, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("train/loss_content_ref", loss_content_ref * self.hparams.lambda_content, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("train/loss_contrastive", loss_ntxent * self.hparams.start_contrastive_weight, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            loss_total = loss_regression_ref_l * self.hparams.lambda_regr_l + \
                         loss_regression_ref_ab * self.hparams.lambda_regr_ab + \
                         loss_recontruction_ref_l * self.hparams.lambda_regr_l * 0.1 + \
                         loss_recontruction_ref_ab * self.hparams.lambda_regr_ab * 0.1 + \
                         loss_ntxent * self.hparams.start_contrastive_weight + \
                         loss_content_ref * self.hparams.lambda_content + loss_style_ref * self.hparams.lambda_style
        else:
            raise NotImplementedError
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
                reconstruction_tgt = TextureMapDataset.apply_mask_texture(self.decoder(self.fold_features(self.fenc_target(self.train_dataset.unfold(batch['texture'])).unsqueeze(-1).unsqueeze(-1))), batch['mask_texture'])
                refinement, refinement_noret, refinement_noinp, s, b = self.attention_blending_decode(batch['mask_texture'], features_in.to(self.device), features_candidates.unsqueeze(1).to(self.device), return_debug_vis=True)

                ds_vis.visualize_texture_batch_01(batch['partial_texture'].cpu().numpy(), batch['texture'].cpu().numpy(), self.fold(batch['database_textures']).unsqueeze(1).cpu().numpy(), refinement_noret.cpu().numpy(),
                                                  refinement_noinp.cpu().numpy(), reconstruction_tgt.cpu().numpy(), refinement.cpu().numpy(),
                                                  (s / 2 + 0.5).cpu().numpy(), ((s / 2 + 0.5) * b).cpu().numpy(), lambda prefix: output_dir / "val_vis" / f"{prefix}_{batch_idx:04d}.jpg")
                total_loss_ref_regression += self.mse_loss(refinement.to(self.device), batch['texture']).cpu().item()

        total_loss_ref_regression /= len(ds_vis)
        self.log("val/loss_ref_regression", total_loss_ref_regression, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True)

    def on_train_start(self):
        self.feature_loss_helper.move_to_device(self.device)

    def on_validation_start(self):
        self.feature_loss_helper.move_to_device(self.device)


class OptimizerChangeCallback(Callback):

    def on_train_epoch_start(self, trainer, pl_module):
        if trainer.current_epoch in pl_module.hparams.phase_epochs and pl_module.current_phase == pl_module.hparams.phase_epochs.index(pl_module.current_epoch):
            pl_module.current_phase += 1
            print("Changing current_phase to:", pl_module.current_phase)
            trainer.optimizers = pl_module.get_current_optimizer()()[0]


@hydra.main(config_path='../config', config_name='texture_end2end_only_refine')
def main(config):
    from datetime import datetime
    import os
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

    trainer = Trainer(gpus=-1, accelerator='ddp', plugins=DDPPlugin(find_unused_parameters=True), num_sanity_val_steps=config.sanity_steps, max_epochs=config.max_epoch, limit_val_batches=config.val_check_percent,
                      callbacks=[checkpoint_callback, OptimizerChangeCallback()], val_check_interval=float(min(config.val_check_interval, 1)), check_val_every_n_epoch=max(1, config.val_check_interval), resume_from_checkpoint=config.resume, logger=logger, benchmark=True)

    model = TextureEnd2EndModule(config)
    trainer.fit(model)


if __name__ == '__main__':
    main()
