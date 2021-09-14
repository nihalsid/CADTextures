from pathlib import Path
from random import randint

import torch
import numpy as np
import hydra
import pytorch_lightning as pl

from PIL import Image

from dataset.texture_patch_dataset import TexturePatchDataset
from model.diffusion_model import EncoderDecoder
from model.discriminator import TextureGANDiscriminatorLocal
from model.texture_gan import TextureGANSlim
from util.gan_loss import GANLoss
from util.misc import resize_npy_as_image, normalize_tensor_color


class PatchOptimizationTrainer(pl.LightningModule):

    def __init__(self, config):
        super(PatchOptimizationTrainer, self).__init__()
        self.save_hyperparameters(config)
        self.discriminator = TextureGANDiscriminatorLocal(3, self.hparams.model.discriminator_ngf)
        dataset = lambda size: TexturePatchDataset(self.hparams, self.hparams.shape, self.hparams.view_index, self.hparams.patch_size, size)
        self.train_dataset, self.val_dataset, self.val_vis_dataset = dataset(self.hparams.batch_size * 400), dataset(self.hparams.batch_size), dataset(self.hparams.batch_size)
        if config.generator == 'cnn':
            self.generator = TextureGANSlim(5, 3, self.hparams.model.input_texture_ngf)
        else:
            self.register_buffer('texture', torch.from_numpy(np.ascontiguousarray(np.transpose(self.train_dataset.partial_texture, (2, 0, 1)))).unsqueeze(0))
            self.texture = normalize_tensor_color(self.texture, config.dataset.color_space)
        self.gan_loss = GANLoss(self.hparams.gan_loss_type)

    def configure_optimizers(self):
        if self.hparams.generator == 'cnn':
            opt_g = torch.optim.Adam(self.generator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        else:
            opt_g = torch.optim.Adam([self.texture.requires_grad_(True)], lr=self.hparams.lr * 100, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=False)

    def forward(self, batch):
        if self.hparams.generator == 'cnn':
            return self.generator(torch.zeros_like(torch.cat([batch['input'], batch['mask_texture'], batch['mask_missing']], 1)))
        return torch.clamp(self.texture.expand((batch['input'].shape[0], -1, -1, -1)), -0.5, 0.5)

    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            self.train_dataset.apply_batch_transforms(batch)
            generated_texture = self.forward(batch)
            generated_patch = self.train_dataset.get_patch_from_tensor(generated_texture)
            g_loss = self.gan_loss.compute_generator_loss(self.discriminator, generated_patch)
            total_loss = g_loss * self.hparams.lambda_g
            self.log(f'adversarial/G', g_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        else:
            self.train_dataset.apply_batch_transforms(batch)
            with torch.no_grad():
                generated_texture = self.forward(batch)
                generated_patch = self.train_dataset.get_patch_from_tensor(generated_texture)
            d_real_loss, d_fake_loss, d_gp_loss = self.gan_loss.compute_discriminator_loss(self.discriminator, batch['input_patch'], generated_patch.detach())
            total_loss = d_real_loss + d_fake_loss + self.hparams.lambda_gp * d_gp_loss
            self.log(f'adversarial/D_real', d_real_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'adversarial/D_fake', d_fake_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'adversarial/D_gp', d_gp_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'adversarial/D', d_real_loss + d_fake_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        return {'loss': total_loss}

    def validation_step(self, batch, batch_idx):
        pass

    def validation_epoch_end(self, outputs):
        self.visualize_outputs()

    def visualize_outputs(self):
        w_img = 128
        vis_image = np.zeros((self.hparams.batch_size * (w_img + 5), 5 * (w_img + 5), 3), dtype=np.uint8)
        output_vis_path = Path("runs") / self.hparams.experiment / f"vis"
        output_vis_path.mkdir(exist_ok=True)
        loader = torch.utils.data.DataLoader(self.val_vis_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False)
        for batch in loader:
            TexturePatchDataset.move_batch_to_gpu(batch, self.device)
            self.train_dataset.apply_batch_transforms(batch)
            generated_texture = self.forward(batch)
            generated_patch = self.train_dataset.get_patch_from_tensor(generated_texture)
            for i in range(self.hparams.batch_size):
                for j, item in enumerate([batch['input'][i], batch['target'][i], generated_texture[i], batch['input_patch'][i], generated_patch[i]]):
                    vis_image[i * (w_img + 5): i * (w_img + 5) + w_img, j * (w_img + 5): j * (w_img + 5) + w_img] = resize_npy_as_image(self.val_vis_dataset.denormalize_and_rgb(item.permute((1, 2, 0)).cpu().numpy()), w_img)
        Image.fromarray(vis_image).save(output_vis_path / f"{self.global_step:08d}.jpg")


@hydra.main(config_path='../config', config_name='gan_optimize')
def main(config):
    from datetime import datetime
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    from util.filesystem_logger import FilesystemLogger

    config.experiment = f"{datetime.now().strftime('%d%m%H%M')}_ganopt_{config['experiment']}"
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)

    seed_everything(config.seed)
    # noinspection PyUnusedLocal
    filesystem_logger = FilesystemLogger(config)

    logger = WandbLogger(project=f'GANOpt', name=config.experiment, id=config.experiment)

    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / config.experiment), filename='_ckpt_{epoch}', save_top_k=-1, verbose=False, every_n_epochs=config.save_epoch)
    model = PatchOptimizationTrainer(config)
    trainer = Trainer(gpus=[0], num_sanity_val_steps=config.sanity_steps, max_epochs=config.max_epoch, limit_val_batches=config.val_check_percent, callbacks=[checkpoint_callback],
                      val_check_interval=float(min(config.val_check_interval, 1)), check_val_every_n_epoch=max(1, config.val_check_interval), resume_from_checkpoint=config.resume, logger=logger, benchmark=True)
    trainer.fit(model)


if __name__ == '__main__':
    main()
