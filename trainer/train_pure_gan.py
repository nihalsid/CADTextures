import itertools
from pathlib import Path
from random import randint

import hydra
import pytorch_lightning as pl
import torch
import numpy as np
from PIL import Image

from dataset.noise_dataset import NoiseDataset
from model.discriminator import TextureGANDiscriminatorSlim
from model.texture_gan import ScribblerGenerator
from util.gan_loss import GANLoss


class GANTrainer(pl.LightningModule):

    def __init__(self, config):
        super(GANTrainer, self).__init__()
        self.preload_dict = {}
        self.hparams = config
        self.z_dim = 128
        self.only_l_channel = config.only_l
        self.generator = ScribblerGenerator(self.z_dim, 3, self.hparams.model.generator_ngf)
        self.discriminator = TextureGANDiscriminatorSlim(1 if self.only_l_channel else 3, self.hparams.model.discriminator_ngf)
        dataset = lambda size: NoiseDataset(self.hparams, self.z_dim, size)
        self.train_dataset, self.val_dataset, self.val_vis_dataset = dataset(10240), dataset(24), dataset(49)
        self.gan_loss = GANLoss(self.hparams.gan_loss_type)
        self.image_side = 7
        self.vis_vector = torch.from_numpy(np.random.normal(0, 1, size=(self.image_side * self.image_side, self.z_dim)).astype(np.float32))
        self.visualize_outputs(True)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.9))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.9))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=False)

    def forward(self, batch):
        preds = self.generator(batch['input'])
        return preds

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            self.train_dataset.apply_batch_transform(batch)
            generated_texture = self.forward(batch)
            g_loss = self.gan_loss.compute_generator_loss(self.discriminator, generated_texture[:, 0: (1 if self.only_l_channel else 3), :, :])
            total_loss = g_loss * self.hparams.lambda_g
            self.log(f'adversarial/G', g_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        else:
            self.train_dataset.apply_batch_transform(batch)
            with torch.no_grad():
                generated_texture = self.forward(batch)
            d_real_loss, d_fake_loss, d_gp_loss = self.gan_loss.compute_discriminator_loss(self.discriminator, batch['target'][:, 0:(1 if self.only_l_channel else 3), :, :],
                                                                                           generated_texture.detach()[:, 0:(1 if self.only_l_channel else 3), :, :])
            total_loss = self.hparams.lambda_d * (d_real_loss + d_fake_loss) + self.hparams.lambda_gp * d_gp_loss * (int(self.hparams.lambda_d > 0))
            self.log(f'adversarial/D_real', d_real_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'adversarial/D_fake', d_fake_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'adversarial/D_gp', d_gp_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'adversarial/D', d_real_loss + d_fake_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        self.visualize_outputs()

    def visualize_outputs(self, target_mode=False):
        w_img = 128
        vis_image = np.zeros((self.image_side * (w_img + 5), self.image_side * (w_img + 5), 3), dtype=np.uint8)
        output_vis_path = Path("runs") / self.hparams.experiment / f"vis"
        output_vis_path.mkdir(exist_ok=True)
        if target_mode:
            vis_dataloader = torch.utils.data.DataLoader(self.val_vis_dataset, batch_size=1, shuffle=False, num_workers=0, drop_last=False)
            for i, batch in enumerate(itertools.islice(vis_dataloader, self.image_side * self.image_side)):
                self.val_vis_dataset.apply_batch_transform(batch)
                _r = i // self.image_side
                _c = i % self.image_side
                vis_image[_r * (w_img + 5): _r * (w_img + 5) + w_img, _c * (w_img + 5): _c * (w_img + 5) + w_img] = self.val_vis_dataset.denormalize_and_rgb(batch['target'].squeeze(0).permute((1, 2, 0)).numpy(), self.only_l_channel)
        else:
            predicted_texture = self.generator(self.vis_vector.cuda(self.device)).cpu()
            for i in range(self.image_side * self.image_side):
                _r = i // self.image_side
                _c = i % self.image_side
                vis_image[_r * (w_img + 5): _r * (w_img + 5) + w_img, _c * (w_img + 5): _c * (w_img + 5) + w_img] = self.val_vis_dataset.denormalize_and_rgb(predicted_texture[i].permute((1, 2, 0)).numpy(), self.only_l_channel)
        Image.fromarray(vis_image).save(output_vis_path / f"{self.global_step:08d}.jpg")


@hydra.main(config_path='../config', config_name='gan_pure')
def main(config):
    from datetime import datetime
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    from util.filesystem_logger import FilesystemLogger

    config.experiment = f"{datetime.now().strftime('%d%m%H%M')}_gantest_{config['experiment']}"
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)

    seed_everything(config.seed)
    # noinspection PyUnusedLocal
    filesystem_logger = FilesystemLogger(config)

    logger = WandbLogger(project=f'GANTest', name=config.experiment, id=config.experiment)

    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / config.experiment), filename='_ckpt_{epoch}', save_top_k=-1, verbose=False, period=config.save_epoch)
    model = GANTrainer(config)
    trainer = Trainer(gpus=[0], num_sanity_val_steps=config.sanity_steps, max_epochs=config.max_epoch, limit_val_batches=config.val_check_percent, callbacks=[checkpoint_callback],
                      val_check_interval=float(min(config.val_check_interval, 1)), check_val_every_n_epoch=max(1, config.val_check_interval), resume_from_checkpoint=config.resume, logger=logger, benchmark=True)
    trainer.fit(model)


if __name__ == '__main__':
    main()
