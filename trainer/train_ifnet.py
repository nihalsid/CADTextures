from random import randint

import hydra
import pytorch_lightning as pl
import torch
from pathlib import Path

from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.utilities import rank_zero_only
import numpy as np

from dataset.ifnet_dataset import ImplicitDataset
from model.ifnet import TEXR


class TextureIFNetModule(pl.LightningModule):

    def __init__(self, config):
        super(TextureIFNetModule, self).__init__()
        self.save_hyperparameters(config)
        self.preload_dict = {}
        assert config.dataset.texture_map_size == 128, "only 128x128 texture map supported"
        self.model = TEXR()
        self.dataset = lambda split: ImplicitDataset(config, split, self.preload_dict)
        self.train_dataset = self.dataset('train')

        x_range = y_range = np.linspace(-1, 1, 128).astype(np.float32)
        grid_x, grid_y = np.meshgrid(x_range, y_range, indexing='ij')
        grid_x, grid_y = grid_x.flatten(), grid_y.flatten()
        self.stacked = torch.from_numpy(np.hstack((grid_x[:, np.newaxis], grid_y[:, np.newaxis]))).to(self.device).unsqueeze(0)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.hparams.lr)
        return [optimizer], []

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, drop_last=True, pin_memory=True)

    def val_dataloader(self):
        dataset = self.dataset('train')
        return torch.utils.data.DataLoader(dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False, pin_memory=True)

    def step(self, batch):
        pred_rgb = self.model(batch['grid_coords'], torch.cat([batch['partial_texture'], batch['mask_missing']], 1))
        pred_rgb = pred_rgb.transpose(-1, -2)
        loss_i = torch.nn.L1Loss(reduction='none')(pred_rgb, batch['values'])
        loss = loss_i.sum(-1).mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self.step(batch)
        self.log("train/loss", loss.item(), on_step=True, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_idx):
        pass

    def visualize_prediction(self, x):
        assert x.shape[0] == 1
        with torch.no_grad():
            rgb = torch.clamp(self.model(self.stacked.to(self.device), x), -0.5, 0.5).detach().cpu()
        image = rgb.squeeze(0).reshape((3, 128, 128)).permute((1, 2, 0)).numpy()
        return image

    @rank_zero_only
    def validation_epoch_end(self, outputs):
        dataset = lambda split: ImplicitDataset(self.hparams, split, self.preload_dict, single_view=True)
        output_dir = Path("runs") / self.hparams.experiment / "visualization" / f"epoch_{self.current_epoch:04d}"
        output_dir.mkdir(exist_ok=True, parents=True)

        ds_vis = dataset('val_vis')

        loader = torch.utils.data.DataLoader(ds_vis, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, drop_last=False, pin_memory=True)

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                gt, pred = [], []
                ds_vis.move_batch_to_gpu(batch, self.device)
                x = torch.cat([batch['partial_texture'], batch['mask_missing']], 1)
                for b in range(x.shape[0]):
                    predicted_image = self.visualize_prediction(x[b: b + 1, :, :, :])
                    pred.append(predicted_image)
                ds_vis.visualize_texture_batch(ds_vis.get_true_texture(batch['name']), pred, batch['mask_texture'].cpu().numpy(), output_dir / f"{batch_idx:04d}.jpg")


@hydra.main(config_path='../config', config_name='texture_ifnet')
def main(config):
    from datetime import datetime
    from pytorch_lightning import Trainer, seed_everything
    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import WandbLogger
    from util.filesystem_logger import FilesystemLogger
    ds_name = '_'.join(config.dataset.name.split('/'))
    if not config.wandb_main and config.suffix == '':
        config.suffix = '-dev'
    config.experiment = f"{datetime.now().strftime('%d%m%H%M')}_ifnet_{config['experiment']}"
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)

    seed_everything(config.seed)
    # noinspection PyUnusedLocal
    filesystem_logger = FilesystemLogger(config)
    logger = WandbLogger(project=f'IFNet{config.suffix}[{ds_name}]', name=config.experiment, id=config.experiment)
    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / config.experiment), filename='_ckpt_{epoch}', save_top_k=-1, verbose=False, period=config.save_epoch)
    model = TextureIFNetModule(config)
    trainer = Trainer(gpus=-1, accelerator='ddp', plugins=DDPPlugin(find_unused_parameters=False), num_sanity_val_steps=config.sanity_steps, max_epochs=config.max_epoch, limit_val_batches=config.val_check_percent, callbacks=[checkpoint_callback],
                      val_check_interval=float(min(config.val_check_interval, 1)), check_val_every_n_epoch=max(1, config.val_check_interval), resume_from_checkpoint=config.resume, logger=logger, benchmark=True)
    trainer.fit(model)


if __name__ == '__main__':
    main()
