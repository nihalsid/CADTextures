from random import randint

import hydra
import pytorch_lightning as pl
import torch
from pathlib import Path
from PIL import Image
import numpy as np

from dataset.texture_map_dataset import TextureMapDataset
from model.retrieval import Patch16
from util import retrieval
from util.contrastive_loss import NTXentLoss
from util.retrieval import RetrievalInterface


class RetrievalTrainingModule(pl.LightningModule):

    def __init__(self, config):
        super(RetrievalTrainingModule, self).__init__()
        self.hparams = config
        self.preload_dict = {}
        assert config.dataset.texture_map_size == 128, "only 128x128 texture map supported"
        self.fenc_input, self.fenc_target = Patch16(config.fenc_nf, config.fenc_zdim), Patch16(config.fenc_nf, config.fenc_zdim)
        self.nt_xent_loss = NTXentLoss(float(config.temperature), True)
        self.current_learning_rate = config.lr
        self.retrieval_handler = RetrievalInterface(config.dictionary, config.fenc_zdim)
        self.dataset = lambda split: TextureMapDataset(config, split, self.preload_dict)
        self.train_dataset = self.dataset('train')

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
        dataset = self.dataset('val')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False, pin_memory=True)

    def forward(self, batch):
        features_in = self.fenc_input(batch['partial_texture'])
        features_tgt = self.fenc_target(batch['texture'])
        return features_in, features_tgt

    def step(self, batch):
        self.train_dataset.apply_batch_transforms(batch)
        features_in, features_tgt = self.forward(batch)
        features_in_reshaped, features_tgt_reshaped = torch.nn.functional.normalize(features_in, dim=1), torch.nn.functional.normalize(features_tgt, dim=1)
        loss_contrastive = self.nt_xent_loss(features_in_reshaped, features_tgt_reshaped)
        return loss_contrastive

    def training_step(self, batch, batch_idx):
        loss_contrastive = self.step(batch)
        self.log("learning_rate", self.current_learning_rate, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        self.log("train/contrastive_loss", loss_contrastive, on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss_contrastive}

    def validation_step(self, batch, batch_idx):
        loss_contrastive = self.step(batch)
        self.log("val/contrastive_loss", loss_contrastive, on_step=False, on_epoch=True, prog_bar=False, logger=True)

    def validation_epoch_end(self, outputs):
        output_dir = Path("runs") / self.hparams.experiment / "visualization" / f"epoch_{self.current_epoch:04d}"
        output_dir.mkdir(exist_ok=True, parents=True)
        (output_dir / "val_vis").mkdir(exist_ok=True)
        ds_train = self.dataset('train')
        ds_val = self.dataset('val')
        ds_vis = self.dataset('val_vis')
        ds_train_eval = self.dataset('train_val')
        retrieval.create_dictionary(self.fenc_target, self.hparams.dictionary, self.hparams.fenc_zdim, ds_train, output_dir)
        ds_train.set_all_view_indexing(True)
        ds_val.set_all_view_indexing(True)
        ds_vis.set_all_view_indexing(True)
        ds_train_eval.set_all_view_indexing(True)
        # remove train eval since it takes too long
        # print('[Eval-Train]')
        # train_eval_retrievals = self.retrieval_handler.create_mapping_and_retrieve_nearest_textures_for_all(self.fenc_input, output_dir, ds_train, ds_train_eval, 1, True)
        # t_err = retrieval.get_error_retrieval(train_eval_retrievals, ds_train_eval)
        # print(f"Train Error: {t_err:.3f}\n")
        # self.log("train/loss_l1", t_err, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        print('[Eval-Train-GT]')
        train_eval_retrievals = self.retrieval_handler.create_mapping_and_retrieve_nearest_textures_for_all(self.fenc_input, output_dir, ds_train, ds_train_eval, 1, False)
        t_err = retrieval.get_error_retrieval(train_eval_retrievals, ds_train_eval)
        print(f"Train-GT Error: {t_err:.3f}\n")
        self.log("traingt/loss_l1", t_err, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        print('[Eval-Validation]')
        val_retrievals = self.retrieval_handler.create_mapping_and_retrieve_nearest_textures_for_all(self.fenc_input, output_dir, ds_train, ds_val, 1, False)
        v_err = retrieval.get_error_retrieval(val_retrievals, ds_val)
        self.log("val/loss_l1", v_err, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        print(f"Val Error: {v_err:.3f}\n")
        vis_retrievals = self.retrieval_handler.create_mapping_and_retrieve_nearest_textures_for_all(self.fenc_input, output_dir, ds_train, ds_vis, 1, False)
        for idx in range(vis_retrievals.shape[0]):
            vis_image = np.ones((128, 2 * 128 + 20, 3), dtype=np.uint8) * 255
            retrieved_texture = ds_vis.convert_data_for_visualization([vis_retrievals[idx, 0].cpu().numpy()], [], [])[0][0]
            item, view_idx = ds_vis.get_item_and_view_idx(idx)
            vis_image[:, :128, :] = (np.clip(ds_vis.to_rgb(np.transpose(ds_vis.get_texture(item), (1, 2, 0))), 0, 255)).astype(np.uint8)
            vis_image[:, 148:, :] = (retrieved_texture * 255).astype(np.uint8)
            Image.fromarray(vis_image).save(output_dir / "val_vis" / f"{item}__{view_idx}.png")


@hydra.main(config_path='../config', config_name='retrieval')
def main(config):
    from datetime import datetime
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    from util.filesystem_logger import FilesystemLogger
    ds_name = '_'.join(config.dataset.name.split('/'))
    if not config.wandb_main and config.suffix == '':
        config.suffix = '-dev'
    config.experiment = f"{datetime.now().strftime('%d%m%H%M')}_retrieval_{config['experiment']}"
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)

    seed_everything(config.seed)
    # noinspection PyUnusedLocal
    filesystem_logger = FilesystemLogger(config)
    logger = WandbLogger(project=f'Retrieval{config.suffix}[{ds_name}]', name=config.experiment, id=config.experiment)
    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / config.experiment), filename='_ckpt_{epoch}', save_top_k=-1, verbose=False, period=config.save_epoch)
    model = RetrievalTrainingModule(config)
    trainer = Trainer(gpus=[0], num_sanity_val_steps=config.sanity_steps, max_epochs=config.max_epoch, limit_val_batches=config.val_check_percent, callbacks=[checkpoint_callback],
                      val_check_interval=float(min(config.val_check_interval, 1)), check_val_every_n_epoch=max(1, config.val_check_interval), resume_from_checkpoint=config.resume, logger=logger, benchmark=True)
    trainer.fit(model)


if __name__ == '__main__':
    main()
