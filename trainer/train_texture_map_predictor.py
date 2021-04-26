from pathlib import Path
from random import randint
import hydra
import pytorch_lightning as pl
import torch
import os
import numpy as np
from PIL import Image
import json

from dataset.texture_map_dataset import TextureMapDataset
from model.scribbler import get_model


class TextureMapPredictorModule(pl.LightningModule):

    def __init__(self, config):
        super(TextureMapPredictorModule, self).__init__()
        self.hparams = config
        dataset = lambda split: TextureMapDataset(config, split)
        self.train_dataset, self.val_dataset, self.train_val_dataset, self.train_vis_dataset, self.val_vis_dataset = dataset('train'), dataset('val'), dataset('train_val'), dataset('train_vis'), dataset('val_vis')
        self.model = get_model(config)

    def forward(self, batch):
        TextureMapDataset.apply_batch_transforms(batch)
        input_maps = batch['mask_texture']
        condition = [torch.cat([batch['render'], batch['mask_render']], dim=1), ]
        if 'noc' in self.hparams.inputs:
            input_maps = torch.cat([input_maps, batch['noc']], dim=1)
        if 'partial_texture' in self.hparams.inputs:
            input_maps = torch.cat([input_maps, batch['partial_texture']], dim=1)
        if 'noc_render' in self.hparams.inputs:
            condition[0] = torch.cat([condition[0], batch['noc_render']], dim=1)
        if 'normal' in self.hparams.inputs:
            input_maps = torch.cat([input_maps, batch['normal']], dim=1)
        if 'distance_field' in self.hparams.inputs:
            condition.append(batch['df'])
        predicted_texture = self.model(input_maps, *condition)
        return predicted_texture

    @staticmethod
    def loss(target, prediction, weights):
        return torch.abs(prediction - target) * weights.expand(-1, target.shape[1], -1, -1)

    def training_step(self, batch, batch_index):
        predicted_texture = self.forward(batch)
        loss = self.loss(batch['texture'], predicted_texture, batch['mask_texture']).mean()
        self.log('train/loss_l1', loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        return {'loss': loss}

    def validation_step(self, batch, batch_index, dataloader_index):
        split = ["val", "train"][dataloader_index]
        suffix = ["", "_epoch"][dataloader_index]
        predicted_texture = self.forward(batch)
        loss = self.loss(batch['texture'], predicted_texture, batch['mask_texture']).mean()
        self.log(f'{split}/loss_l1{suffix}', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

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
                    TextureMapDataset.move_batch_to_gpu(batch, self.device)
                    predicted_texture = self.forward(batch)
                    loss = self.loss(batch['texture'], predicted_texture, batch['mask_texture']).mean(axis=1).squeeze(1)
                    for ii in range(predicted_texture.shape[0]):
                        self.visualize_prediction(output_vis_path, batch['name'][ii], batch['view_index'][ii], batch['texture'][ii].cpu().numpy(), batch['render'][ii].cpu().numpy(), batch['partial_texture'][ii].cpu().numpy(), predicted_texture[ii].cpu().numpy(), loss[ii].cpu().numpy())

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.hparams.lr)
        return [optimizer]

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return [torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=False),
                torch.utils.data.DataLoader(self.train_val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=False)]

    def visualize_prediction(self, save_dir, name, v_idx, texture, render, partial_texture, prediction, loss):
        import matplotlib.pyplot as plt
        texture = np.transpose(texture, (1, 2, 0))
        prediction = np.transpose(prediction, (1, 2, 0))
        render = np.transpose(render, (1, 2, 0))
        partial_texture = np.transpose(partial_texture, (1, 2, 0))
        loss = (loss - loss.min()) / (loss.max() - loss.min())
        f, axarr = plt.subplots(1, 6, figsize=(4, 6))
        axarr[0].imshow(render + 0.5)
        axarr[0].axis('off')
        axarr[1].imshow(partial_texture + 0.5)
        axarr[1].axis('off')
        axarr[2].imshow(texture + 0.5)
        axarr[2].axis('off')
        axarr[3].imshow(prediction + 0.5)
        axarr[3].axis('off')
        axarr[4].imshow(loss, cmap='jet')
        axarr[4].axis('off')
        closest_plotted = False
        closest_train = Path(self.hparams.dataset.data_dir) / 'splits' / self.hparams.dataset.name / 'closest_train.json'
        if closest_train.exists():
            closest_train_dict = json.loads(closest_train.read_text())
            if name in closest_train_dict:
                texture_path = self.train_dataset.path_to_dataset / closest_train_dict[name] / "surface_texture.png"
                if texture_path.exists():
                    with Image.open(texture_path) as texture_im:
                        closest = TextureMapDataset.process_to_padded_thumbnail(texture_im, self.train_dataset.texture_map_size) / 255 - 0.5
                    axarr[5].imshow(closest + 0.5)
                    axarr[5].axis('off')
                    closest_plotted = True
        if not closest_plotted:
            axarr[4].imshow(np.zeros_like(loss), cmap='binary')
            axarr[4].axis('off')
        plt.savefig(save_dir / "figures" / f"{name}_{v_idx}.jpg", bbox_inches='tight', dpi=240)
        plt.close()
        obj_text = Path(self.hparams.dataset.data_dir, self.hparams.dataset.mesh_dir, name, "normalized_model.obj").read_text()
        obj_text = "\n".join([x for x in obj_text.splitlines() if x.split(' ')[0] not in ('mtllib', 'usemtl')])
        gt_obj_text = f"mtllib {name}_gt.mtl\nusemtl material\n{obj_text}"
        pred_obj_text = f"mtllib {name}_pred.mtl\nusemtl material\n{obj_text}"
        dummy_mtl_text = "newmtl material\nKd 1 1 1\nKa 0.1 0.1 0.1\nKs 0.4 0.4 0.4\nKe 0 0 0\nNs 10\nillum 2\nmap_Kd "
        gt_mtl_text = dummy_mtl_text + f"{name}_gt.jpg"
        pred_mtl_text = dummy_mtl_text + f"{name}_pred.jpg"
        Path(save_dir / "meshes" / f"{name}_gt.obj").write_text(gt_obj_text)
        Path(save_dir / "meshes" / f"{name}_pred.obj").write_text(pred_obj_text)
        Path(save_dir / "meshes" / f"{name}_gt.mtl").write_text(gt_mtl_text)
        Path(save_dir / "meshes" / f"{name}_pred.mtl").write_text(pred_mtl_text)
        Image.fromarray(((texture + 0.5) * 255).astype(np.uint8)).save(save_dir / "meshes" / f"{name}_gt.jpg")
        Image.fromarray(((prediction + 0.5) * 255).astype(np.uint8)).save(save_dir / "meshes" / f"{name}_pred.jpg")


@hydra.main(config_path='../config', config_name='base')
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
    logger = WandbLogger(project=f'CADTextures{config.suffix}[{ds_name}]', name=config.experiment, id=config.experiment)

    checkpoint_callback = ModelCheckpoint(dirpath=(Path("runs") / config.experiment), filename='_ckpt_{epoch}', save_top_k=-1, verbose=False, period=config.save_epoch)
    model = TextureMapPredictorModule(config)
    trainer = Trainer(gpus=[0], num_sanity_val_steps=config.sanity_steps, max_epochs=config.max_epoch, limit_val_batches=config.val_check_percent, callbacks=[checkpoint_callback],
                      val_check_interval=float(min(config.val_check_interval, 1)), check_val_every_n_epoch=max(1, config.val_check_interval), resume_from_checkpoint=config.resume, logger=logger, benchmark=True)
    trainer.fit(model)


if __name__ == '__main__':
    main()
