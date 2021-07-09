from random import randint

import hydra
import pytorch_lightning as pl
import torch
from pathlib import Path
import json
from PIL import Image
import os
import numpy as np

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only

from dataset.texture_end2end_dataset import TextureEnd2EndDataset
from dataset.texture_map_dataset import TextureMapDataset
from model.fold import Fold2D
from model.refinement import Decoder8x8
from model.retrieval import get_target_feature_extractor, get_input_feature_extractor, get_norm
from trainer.train_texture_map_predictor import TextureMapPredictorModule
from util.contrastive_loss import NTXentLoss
from util.feature_loss import FeatureLossHelper
from util.regression_loss import RegressionLossHelper


class TextureEnd2EndModule(pl.LightningModule):

    def __init__(self, config):
        super(TextureEnd2EndModule, self).__init__()
        self.save_hyperparameters(config)
        self.preload_dict = {}
        self.prefc_features = 384
        assert config.dataset.texture_map_size == 128, "only 128x128 texture map supported"
        self.fenc_input, self.fenc_target = get_input_feature_extractor(config), get_target_feature_extractor(config)
        self.current_learning_rate = config.lr
        self.nt_xent_loss = NTXentLoss(float(config.temperature), config.dataset.texture_map_size // config.dictionary.patch_size, True)
        self.regression_loss = RegressionLossHelper(self.hparams.regression_loss_type)
        self.feature_loss_helper = FeatureLossHelper(['relu4_2'], ['relu3_2', 'relu4_2'])
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.start_contrastive_weight = config.start_contrastive_weight
        self.current_contrastive_weight = self.start_contrastive_weight
        self.dataset = lambda split: TextureEnd2EndDataset(config, split, self.preload_dict)
        self.train_dataset = self.dataset('train')
        self.fold = Fold2D(config.dataset.texture_map_size // config.dictionary.patch_size, config.dictionary.patch_size, 3)
        self.fold_features = Fold2D(config.dataset.texture_map_size // config.dictionary.patch_size, 1, self.prefc_features * 2)
        self.fold_s = Fold2D(config.dataset.texture_map_size // config.dictionary.patch_size, 1, config.dictionary.K)
        self.decoder = Decoder8x8([self.prefc_features * 2, 384, 160, 64, 32, 16, 3], get_norm(config.norm))
        self.attention = torch.nn.MultiheadAttention(embed_dim=384, num_heads=1, batch_first=True)
        self.current_phase = config.current_phase

    def on_train_epoch_start(self):
        if self.current_epoch in self.hparams.phase_epochs and self.current_phase == self.hparams.phase_epochs.index(self.current_epoch):
            self.current_phase += 1
        self.current_contrastive_weight = self.start_contrastive_weight * max(0, self.hparams.warmup_epochs_constrastive - self.current_epoch) / self.hparams.warmup_epochs_constrastive

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.fenc_input.parameters()) + list(self.fenc_target.parameters()), lr=self.hparams.lr)
        scheduler = []
        if self.hparams.scheduler is not None:
            scheduler = [torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.scheduler, gamma=0.5)]
        return [optimizer], scheduler

    # noinspection PyMethodOverriding
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_idx, optimizer_closure, on_tpu=False, using_native_amp=False, using_lbfgs=False):
        # warm up lr
        if self.trainer.global_step < 1500 and self.hparams.scheduler is not None:
            lr_scale = min(1., float(self.trainer.global_step + 1) / 1500.)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * self.hparams.lr
        # update params
        self.current_learning_rate = optimizer.param_groups[0]['lr']
        optimizer.step(closure=optimizer_closure)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        dataset = self.dataset('train')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False, pin_memory=True)

    def get_features(self, batch, _use_argmax=False):
        features_in, features_in_prefc = self.fenc_input(batch['partial_texture'], batch['mask_missing'])
        features_tgt_normed = torch.nn.functional.normalize(self.fenc_target(self.train_dataset.unfold(batch['texture']))[0], dim=1)
        features_candidates, features_candidates_prefc = self.fenc_target(batch['database_textures'])
        features_in_normed, features_candidates_normed = torch.nn.functional.normalize(features_in, dim=1), torch.nn.functional.normalize(features_candidates, dim=1)
        features_candidates_normed = features_candidates_normed.unsqueeze(0).expand(features_in_normed.shape[0], -1, -1)
        features_candidates_prefc = features_candidates_prefc.unsqueeze(0).expand(features_in_normed.shape[0], -1, -1)
        return features_in_normed, features_in_prefc, features_tgt_normed, features_candidates_normed, features_candidates_prefc

    def forward_retrieval(self, batch):
        features_in_normed, features_in_prefc, features_tgt_normed, features_candidates_normed, features_candidates_prefc = self.get_features(batch)
        similarity_scores = torch.einsum('ik,ijk->ij', features_in_normed, features_candidates_normed)
        candidates = batch['database_textures'].unsqueeze(0).expand(similarity_scores.shape[0], -1, -1, -1, -1).view(similarity_scores.shape[0], similarity_scores.shape[1], -1)
        scaled_similarity_scores = similarity_scores * 25
        selection_mask = torch.nn.functional.gumbel_softmax(scaled_similarity_scores, tau=1, hard=True)
        selected_patches = torch.einsum('ij,ijk->ik', selection_mask, candidates)
        selected_patches = selected_patches.view(similarity_scores.shape[0], batch['database_textures'].shape[1], batch['database_textures'].shape[2], batch['database_textures'].shape[3])
        return TextureMapDataset.apply_mask_texture(self.fold(selected_patches), batch['mask_texture']), features_in_normed, features_tgt_normed

    def forward_all(self, batch):
        features_in_normed, features_in_prefc, features_tgt_normed, features_candidates_normed, features_candidates_prefc = self.get_features(batch)
        similarity_scores = torch.einsum('ik,ijk->ij', features_in_normed, features_candidates_normed)
        candidates = batch['database_textures'].unsqueeze(0).expand(similarity_scores.shape[0], -1, -1, -1, -1).view(similarity_scores.shape[0], similarity_scores.shape[1], -1)

        scaled_similarity_scores = similarity_scores * 25
        features_subset = features_candidates_normed
        features_subset_prefc = features_candidates_prefc
        selection_mask = torch.nn.functional.gumbel_softmax(scaled_similarity_scores, tau=1, hard=True)
        selected_patches = torch.einsum('ij,ijk->ik', selection_mask, candidates)
        selected_patches = selected_patches.view(similarity_scores.shape[0], batch['database_textures'].shape[1], batch['database_textures'].shape[2], batch['database_textures'].shape[3])
        retrieval = TextureMapDataset.apply_mask_texture(self.fold(selected_patches), batch['mask_texture'])

        knn_candidate_features_prefc = []
        knn_candidate_features_normed = []

        for k in range(self.hparams.dictionary.K - 1):
            selected_features_prefc = torch.einsum('ij,ijk->ik', selection_mask, features_subset_prefc)
            selected_features = torch.einsum('ij,ijk->ik', selection_mask, features_subset)
            knn_candidate_features_prefc.append(selected_features_prefc.unsqueeze(1))
            knn_candidate_features_normed.append(selected_features.unsqueeze(1))
            mask = (torch.ones_like(selection_mask) - selection_mask).bool()
            scaled_similarity_scores = scaled_similarity_scores[mask].reshape((selection_mask.shape[0], selection_mask.shape[1] - 1))
            features_subset_prefc = features_subset_prefc[mask, :].reshape((features_subset_prefc.shape[0], features_subset_prefc.shape[1] - 1, features_subset_prefc.shape[2]))
            features_subset = features_subset[mask, :].reshape((features_subset.shape[0], features_subset.shape[1] - 1, features_subset.shape[2]))
            selection_mask = torch.nn.functional.gumbel_softmax(scaled_similarity_scores, tau=1, hard=True)

        # TODO: optimize this by saving the sampling locations in a cache for each item + view_idx
        source_candidates = TextureMapDataset.sample_patches(1 - batch['mask_missing'], self.hparams.dictionary.patch_size, 256, batch['partial_texture'])[0]
        features_source_candidates_normed, features_source_candidates_prefc = TextureEnd2EndDataset.get_texture_patch_codes(self.fenc_target, source_candidates, self.fold_features.num_patch_x ** 2, self.device, self.device)
        scaled_similarity_scores_source = torch.einsum('ik,ijk->ij', features_in_normed, features_source_candidates_normed) * 25
        selections_source_mask = torch.nn.functional.gumbel_softmax(scaled_similarity_scores_source, tau=1, hard=True)
        selected_source_features_prefc = torch.einsum('ij,ijk->ik', selections_source_mask, features_source_candidates_prefc)
        selected_source_features = torch.einsum('ij,ijk->ik', selections_source_mask, features_source_candidates_normed)
        knn_candidate_features_prefc.append(selected_source_features_prefc.unsqueeze(1))
        knn_candidate_features_normed.append(selected_source_features.unsqueeze(1))

        knn_candidate_features_prefc = torch.cat(knn_candidate_features_prefc, 1)
        knn_candidate_features_normed = torch.cat(knn_candidate_features_normed, 1)

        refinement, s, b = self.attention_blending_decode(batch['mask_texture'], features_in_prefc, knn_candidate_features_prefc)
        return retrieval, refinement, features_in_normed, features_tgt_normed, s, b

    def attention_blending_decode(self, mask_texture, features_in_prefc, knn_candidate_features_prefc):
        attn_output, attn_weights = self.attention(features_in_prefc.unsqueeze(1), knn_candidate_features_prefc, knn_candidate_features_prefc)
        attn_output, attn_weights = attn_output.squeeze(1), attn_weights.squeeze(1)
        o = self.fold_features(torch.cat([features_in_prefc, attn_output], 1).view(attn_output.shape[0], features_in_prefc.shape[1] + attn_output.shape[1], 1, 1))
        refinement = TextureMapDataset.apply_mask_texture(self.decoder(o), mask_texture)
        return refinement, self.fold_s(attn_weights.unsqueeze(-1).unsqueeze(-1)), self.fold_s(attn_weights.unsqueeze(-1).unsqueeze(-1))

    def training_step(self, batch, batch_idx):
        self.train_dataset.add_database_to_batch(self.hparams.batch_size * self.hparams.dataset.num_database_textures, batch, self.device, self.hparams.dictionary.patch_size == 128)
        self.train_dataset.apply_batch_transforms(batch)
        gt_texture_l, gt_texture_ab = TextureMapPredictorModule.split_into_channels(batch['texture'])
        loss_total = torch.zeros([1, ], device=self.device)
        if self.current_phase == 0:  # retrieval
            retrieval, features_in, features_tgt = self.forward_retrieval(batch)
            retrieved_texture_l, retrieved_texture_ab = TextureMapPredictorModule.split_into_channels(retrieval)
            loss_regression_ret_l = self.regression_loss.calculate_loss(gt_texture_l, retrieved_texture_l).mean()
            loss_regression_ret_ab = self.regression_loss.calculate_loss(gt_texture_ab, retrieved_texture_ab).mean()
            loss_content_ret = self.feature_loss_helper.calculate_feature_loss(gt_texture_l, retrieved_texture_l).mean()
            style_loss_maps = self.feature_loss_helper.calculate_style_loss(gt_texture_l, retrieved_texture_l)
            loss_style_ret = style_loss_maps[0].mean() + style_loss_maps[1].mean()
            loss_ntxent = self.nt_xent_loss(features_in, features_tgt)
            self.log("train/loss_contrastive", loss_ntxent * self.start_contrastive_weight, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("train/contrastive_weight", self.current_contrastive_weight, on_step=False, on_epoch=True, prog_bar=False, logger=True)
            loss_total = loss_total * 0.5 + (
                        loss_regression_ret_l * self.hparams.lambda_regr_l + loss_regression_ret_ab * self.hparams.lambda_regr_ab + loss_content_ret * self.hparams.lambda_content + loss_style_ret * self.hparams.lambda_style) * 1 \
                         + loss_ntxent * self.current_contrastive_weight
        if self.current_phase == 1:  # refinement
            retrieval, refinement, _, _, score, blend = self.forward_all(batch)
            retrieved_texture_l, retrieved_texture_ab = TextureMapPredictorModule.split_into_channels(retrieval)
            refined_texture_l, refined_texture_ab = TextureMapPredictorModule.split_into_channels(refinement)
            loss_regression_ret_l = self.regression_loss.calculate_loss(gt_texture_l, retrieved_texture_l).mean()
            loss_regression_ret_ab = self.regression_loss.calculate_loss(gt_texture_ab, retrieved_texture_ab).mean()
            loss_content_ret = self.feature_loss_helper.calculate_feature_loss(gt_texture_l, retrieved_texture_l).mean()
            style_loss_maps = self.feature_loss_helper.calculate_style_loss(gt_texture_l, retrieved_texture_l)
            loss_style_ret = style_loss_maps[0].mean() + style_loss_maps[1].mean()
            loss_regression_ref_l = self.regression_loss.calculate_loss(gt_texture_l, refined_texture_l).mean()
            loss_regression_ref_ab = self.regression_loss.calculate_loss(gt_texture_ab, refined_texture_ab).mean()
            loss_content_ref = self.feature_loss_helper.calculate_feature_loss(gt_texture_l, refined_texture_l).mean()
            style_loss_maps = self.feature_loss_helper.calculate_style_loss(gt_texture_l, refined_texture_l)
            loss_style_ref = style_loss_maps[0].mean() + style_loss_maps[1].mean()
            self.log("train/loss_regression_ref_l", loss_regression_ref_l * self.hparams.lambda_regr_l, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("train/loss_regression_ref_ab", loss_regression_ref_ab * self.hparams.lambda_regr_ab, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("train/loss_style_ref", loss_style_ref * self.hparams.lambda_style, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            self.log("train/loss_content_ref", loss_content_ref * self.hparams.lambda_content, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
            loss_total = loss_total + \
                         (loss_regression_ret_l * self.hparams.lambda_regr_l + loss_regression_ret_ab * self.hparams.lambda_regr_ab + loss_content_ret * self.hparams.lambda_content + loss_style_ret * self.hparams.lambda_style) * 0.5 + \
                         (loss_regression_ref_l * self.hparams.lambda_regr_l + loss_regression_ref_ab * self.hparams.lambda_regr_ab + loss_content_ref * self.hparams.lambda_content + loss_style_ref * self.hparams.lambda_style / 25)
        self.log("train/loss_regression_ret_l", loss_regression_ret_l * self.hparams.lambda_regr_l, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/loss_regression_ret_ab", loss_regression_ret_ab * self.hparams.lambda_regr_ab, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/loss_style_ret", loss_style_ret * self.hparams.lambda_style, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log("train/loss_content_ret", loss_content_ret * self.hparams.lambda_content, on_step=True, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
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
        total_loss_ret_regression = 0
        total_loss_ref_regression = 0
        total_loss_contrastive = 0
        with torch.no_grad():
            candidate_codes, candidate_feats_prefc = ds_vis.get_all_texture_patch_codes(self.fenc_target, self.device, self.hparams.batch_size)
            for batch_idx, batch in enumerate(loader):
                ds_vis.move_batch_to_gpu(batch, self.device)
                ds_vis.apply_batch_transforms(batch)

                features_in_unnormed, features_in_prefc = self.fenc_input(batch['partial_texture'], batch['mask_missing'])
                features_in = torch.nn.functional.normalize(features_in_unnormed, dim=1).cpu()
                features_tgt_unnormed, _ = self.fenc_target(ds_vis.unfold(batch['texture']))
                features_tgt = torch.nn.functional.normalize(features_tgt_unnormed, dim=1)
                features_candidates = candidate_codes.unsqueeze(0).expand(features_in.shape[0], -1, -1)
                features_candidates_prefc = candidate_feats_prefc.unsqueeze(0).expand(features_in.shape[0], -1, -1)

                num_patches_x = self.hparams.dataset.texture_map_size // self.hparams.dictionary.patch_size
                source_candidates = TextureMapDataset.sample_patches(1 - batch['mask_missing'], self.hparams.dictionary.patch_size, -1, batch['partial_texture'])[0]
                features_source_candidates, features_source_candidates_prefc = TextureEnd2EndDataset.get_texture_patch_codes(self.fenc_target, source_candidates, num_patches_x * num_patches_x, self.device, torch.device('cpu'))

                selections = torch.argsort(torch.einsum('ik,ijk->ij', features_in, features_candidates), dim=1, descending=True)[:, :self.hparams.dictionary.K - 1]
                retrieved_textures = []
                retrieved_features_prefc = []
                retrieved_features = []
                for k in range(self.hparams.dictionary.K - 1):
                    retrieved_texture = ds_vis.get_patches_with_indices(selections[:, k])
                    retrieved_texture = TextureMapDataset.apply_mask_texture(self.fold(retrieved_texture), batch['mask_texture'].cpu())
                    retrieved_textures.append(retrieved_texture.unsqueeze(1))
                    retrieved_features_prefc.append(features_candidates_prefc[list(range(selections.shape[0])), selections[:, k], :].unsqueeze(1))
                    retrieved_features.append(features_candidates[list(range(selections.shape[0])), selections[:, k], :].unsqueeze(1))

                selections_source = torch.argmax(torch.einsum('ik,ijk->ij', features_in, features_source_candidates), dim=1)
                source_candidates = source_candidates.unsqueeze(1).expand(-1, num_patches_x ** 2, -1, -1, -1, -1).reshape(-1, source_candidates.shape[1], source_candidates.shape[2], source_candidates.shape[3], source_candidates.shape[4])
                retrieved_source_texture = self.fold(source_candidates[list(range(source_candidates.shape[0])), selections_source, :, :, :])
                retrieved_source_texture = TextureMapDataset.apply_mask_texture(retrieved_source_texture, batch['mask_texture'].cpu())
                retrieved_source_features = features_source_candidates[list(range(source_candidates.shape[0])), selections_source, :]
                retrieved_source_features_prefc = features_source_candidates_prefc[list(range(source_candidates.shape[0])), selections_source, :]
                retrieved_textures.append(retrieved_source_texture.unsqueeze(1))
                retrieved_features_prefc.append(retrieved_source_features_prefc.unsqueeze(1))
                retrieved_features.append(retrieved_source_features.unsqueeze(1))

                knn_candidate_features_prefc = torch.cat(retrieved_features_prefc, 1)
                knn_textures = torch.cat(retrieved_textures, 1)

                refinement, s, b = self.attention_blending_decode(batch['mask_texture'], features_in_prefc.to(self.device), knn_candidate_features_prefc.to(self.device))

                ds_vis.visualize_texture_batch_01(batch['partial_texture'].cpu().numpy(), batch['texture'].cpu().numpy(), knn_textures.cpu().numpy(), np.zeros_like(refinement.cpu().numpy()), np.zeros_like(refinement.cpu().numpy()), refinement.cpu().numpy(),
                                                  (s / 2 + 0.5).cpu().numpy(), (s / 2 + 0.5).cpu().numpy(), lambda prefix: output_dir / "val_vis" / f"{prefix}_{batch_idx:04d}.jpg")
                total_loss_ret_regression += self.mse_loss(knn_textures[:, 0, :, :, :].to(self.device), batch['texture']).cpu().item()
                total_loss_ref_regression += self.mse_loss(refinement.to(self.device), batch['texture']).cpu().item()
                total_loss_contrastive += self.nt_xent_loss(features_in.to(self.device), features_tgt).cpu().item()

        total_loss_ret_regression /= len(ds_vis)
        total_loss_ref_regression /= len(ds_vis)
        total_loss_contrastive /= len(ds_vis)
        self.log("val/loss_ret_regression", total_loss_ret_regression, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/loss_ref_regression", total_loss_ref_regression, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/loss_contrastive", total_loss_contrastive, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def on_train_start(self):
        self.feature_loss_helper.move_to_device(self.device)


@hydra.main(config_path='../config', config_name='texture_end2end_attn')
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
    model = TextureEnd2EndModule(config)
    trainer = Trainer(gpus=-1, accelerator='ddp', plugins=DDPPlugin(find_unused_parameters=True), num_sanity_val_steps=config.sanity_steps, max_epochs=config.max_epoch, limit_val_batches=config.val_check_percent,
                      callbacks=[checkpoint_callback],
                      val_check_interval=float(min(config.val_check_interval, 1)), check_val_every_n_epoch=max(1, config.val_check_interval), resume_from_checkpoint=config.resume, logger=logger, benchmark=True)
    trainer.fit(model)


if __name__ == '__main__':
    main()
