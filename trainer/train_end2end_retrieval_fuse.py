from random import randint

import hydra
import pytorch_lightning as pl
import torch
from pathlib import Path

from dataset.texture_end2end_dataset import TextureEnd2EndDataset
from dataset.texture_map_dataset import TextureMapDataset
from model.attention import Fold2D
from model.retrieval import Patch16, Patch16MLP
from util.contrastive_loss import NTXentLoss


class TextureEnd2EndModule(pl.LightningModule):

    def __init__(self, config):
        super(TextureEnd2EndModule, self).__init__()
        self.save_hyperparameters(config)
        self.preload_dict = {}
        assert config.dataset.texture_map_size == 128, "only 128x128 texture map supported"
        self.fenc_input, self.fenc_target = Patch16(config.fenc_nf, config.fenc_zdim), Patch16MLP(config.fenc_nf, config.fenc_zdim)
        self.current_learning_rate = config.lr
        self.nt_xent_loss = NTXentLoss(float(config.temperature), config.dataset.texture_map_size // config.dictionary.patch_size, True)
        self.mse_loss = torch.nn.MSELoss(reduction='mean')
        self.current_contrastive_weight = 1
        self.dataset = lambda split: TextureEnd2EndDataset(config, split, self.preload_dict)
        self.train_dataset = self.dataset('train')
        self.fold = Fold2D(config.dataset.texture_map_size // config.dictionary.patch_size, config.dictionary.patch_size, 3)

    def on_train_epoch_start(self):
        self.current_contrastive_weight = max(0, self.hparams.warmup_epochs_constrastive - self.current_epoch) / self.hparams.warmup_epochs_constrastive

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

    def step(self, batch, use_argmax=False):
        self.train_dataset.apply_batch_transforms(batch)
        features_in = self.fenc_input(batch['partial_texture'])
        features_tgt_normed = torch.nn.functional.normalize(self.fenc_target(self.train_dataset.unfold(batch['texture'])), dim=1)
        features_candidates = self.fenc_target(batch['database_textures'])
        features_in_normed, features_candidates_normed = torch.nn.functional.normalize(features_in, dim=1), torch.nn.functional.normalize(features_candidates, dim=1)
        features_candidates_normed = features_candidates_normed.unsqueeze(0).expand(features_in_normed.shape[0], -1, -1)
        similarity_scores = torch.einsum('ik,ijk->ij', features_in_normed, features_candidates_normed)
        candidates = batch['database_textures'].unsqueeze(0).expand(similarity_scores.shape[0], -1, -1, -1, -1).view(similarity_scores.shape[0], similarity_scores.shape[1], -1)
        if use_argmax:
            selected_patches = torch.einsum('ij,ijk->ik', torch.nn.functional.one_hot(similarity_scores.argmax(dim=1), num_classes=similarity_scores.shape[1]).float().to(self.device), candidates)
        else:
            scaled_similarity_scores = similarity_scores * 10
            selection_mask = torch.nn.functional.gumbel_softmax(scaled_similarity_scores, tau=1, hard=True)
            selected_patches = torch.einsum('ij,ijk->ik', selection_mask, candidates)
        selected_patches = selected_patches.view(similarity_scores.shape[0], batch['database_textures'].shape[1], batch['database_textures'].shape[2], batch['database_textures'].shape[3])
        return TextureMapDataset.apply_mask_texture(self.fold(selected_patches), batch['mask_texture']), features_in_normed, features_tgt_normed

    def training_step(self, batch, batch_idx):
        retrieved_texture, features_in, features_tgt = self.step(batch)
        loss_regression = self.mse_loss(retrieved_texture, batch['texture'])
        loss_ntxent = self.nt_xent_loss(features_in, features_tgt)
        loss_total = loss_ntxent * self.current_contrastive_weight + loss_regression
        self.log("learning_rate", self.current_learning_rate, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("train/loss_regression", loss_regression, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/loss_contrastive", loss_ntxent, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        self.log("train/contrastive_weight", self.current_contrastive_weight, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss_total}

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        dataset = lambda split, load_db: TextureEnd2EndDataset(self.hparams, split, self.preload_dict, single_view=True, load_database=load_db)
        output_dir = Path("runs") / self.hparams.experiment / "visualization" / f"epoch_{self.current_epoch:04d}"
        output_dir.mkdir(exist_ok=True, parents=True)
        ds_train = dataset('train', True)

        (output_dir / "val_vis").mkdir(exist_ok=True)
        ds_vis = dataset('val_vis', False)
        loader = torch.utils.data.DataLoader(ds_vis, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False, pin_memory=True)
        total_loss_regression = 0
        total_loss_contrastive = 0
        with torch.no_grad():
            candidate_codes = ds_vis.get_all_texture_patch_codes(self.fenc_target, self.device, self.hparams.batch_size)
            for batch_idx, batch in enumerate(loader):
                ds_vis.move_batch_to_gpu(batch, self.device)
                ds_vis.apply_batch_transforms(batch)
                features_in = torch.nn.functional.normalize(self.fenc_input(batch['partial_texture']), dim=1).cpu()
                features_tgt = torch.nn.functional.normalize(self.fenc_target(ds_vis.unfold(batch['texture'])), dim=1)
                features_candidates = candidate_codes.unsqueeze(0).expand(features_in.shape[0], -1, -1)
                selections = torch.argmax(torch.einsum('ik,ijk->ij', features_in, features_candidates), dim=1)
                retrieved_texture = ds_vis.get_patches_with_indices(selections)
                retrieved_texture = TextureMapDataset.apply_mask_texture(self.fold(retrieved_texture), batch['mask_texture'].cpu())
                ds_train.visualize_texture_batch(torch.cat([batch['texture'].cpu(), retrieved_texture]).numpy(), output_dir / "val_vis" / f"{batch_idx:04d}.jpg")
                total_loss_regression += self.mse_loss(retrieved_texture.to(self.device), batch['texture']).cpu().item()
                total_loss_contrastive += self.nt_xent_loss(features_in.to(self.device), features_tgt).cpu().item()
        total_loss_regression /= len(ds_vis)
        total_loss_contrastive /= len(ds_vis)
        self.log("val/loss_regression", total_loss_regression, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        self.log("val/loss_contrastive", total_loss_contrastive, on_step=False, on_epoch=True, prog_bar=False, logger=True)

        loader = torch.utils.data.DataLoader(ds_train, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False, pin_memory=True)
        batch = next(iter(loader))
        ds_train.move_batch_to_gpu(batch, self.device)
        with torch.no_grad():
            retrieved_texture, _, _ = self.step(batch, use_argmax=True)
        ds_train.visualize_texture_batch(torch.cat([batch['texture'], retrieved_texture]).cpu().numpy(), output_dir / "train_batch.jpg")


@hydra.main(config_path='../config', config_name='texture_end2end')
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
    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / config.experiment), filename='_ckpt_{epoch}', save_top_k=-1, verbose=False, period=config.save_epoch)
    model = TextureEnd2EndModule(config)
    trainer = Trainer(gpus=[0], num_sanity_val_steps=config.sanity_steps, max_epochs=config.max_epoch, limit_val_batches=config.val_check_percent, callbacks=[checkpoint_callback],
                      val_check_interval=float(min(config.val_check_interval, 1)), check_val_every_n_epoch=max(1, config.val_check_interval), resume_from_checkpoint=config.resume, logger=logger, benchmark=True)
    trainer.fit(model)


if __name__ == '__main__':
    main()
