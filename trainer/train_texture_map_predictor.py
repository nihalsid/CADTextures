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
from model.discriminator import get_discriminator, get_discriminator_local
from model.texture_gan import get_model
from util.feature_loss import FeatureLossHelper
from util.gan_loss import GANLoss
from util.regression_loss import RegressionLossHelper


class TextureMapPredictorModule(pl.LightningModule):

    def __init__(self, config):
        print(os.getcwd())
        super(TextureMapPredictorModule, self).__init__()
        self.hparams = config
        self.preload_dict = {}
        dataset = lambda split: TextureMapDataset(config, split, self.preload_dict)
        self.train_dataset, self.val_dataset, self.train_val_dataset, self.train_vis_dataset, self.val_vis_dataset = dataset('train'), dataset('val'), dataset('train_val'), dataset('train_vis'), dataset('val_vis')
        self.model = get_model(config)
        self.discriminator = get_discriminator(config)
        self.discriminator_local = get_discriminator_local(config)
        self.gan_loss = GANLoss(self.hparams.gan_loss_type)
        self.regression_loss = RegressionLossHelper(self.hparams.regression_loss_type)
        self.feature_loss_helper = FeatureLossHelper(['relu4_2'], ['relu3_2', 'relu4_2'])

    def forward(self, batch):
        self.train_dataset.apply_batch_transforms(batch)
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
        generated_texture = self.model(input_maps, *condition)
        return TextureMapDataset.apply_mask_texture(generated_texture, batch['mask_texture'])

    @staticmethod
    def split_into_channels(tensor):
        tensor_0, tensor_1, tensor_2 = torch.chunk(tensor, 3, dim=1)
        tensor_12 = torch.cat((tensor_1, tensor_2), dim=1)
        return tensor_0, tensor_12

    def calculate_losses(self, generated_texture, batch):
        gt_texture_l, gt_texture_ab = self.split_into_channels(batch['texture'])
        generated_texture_l, generated_texture_ab = self.split_into_channels(generated_texture)
        loss_regression_l = self.regression_loss.calculate_loss(gt_texture_l, generated_texture_l).mean()
        loss_regression_ab = self.regression_loss.calculate_loss(gt_texture_ab, generated_texture_ab).mean()
        loss_content = self.feature_loss_helper.calculate_feature_loss(gt_texture_l, generated_texture_l).mean()
        style_loss_maps = self.feature_loss_helper.calculate_style_loss(gt_texture_l, generated_texture_l)
        loss_style = style_loss_maps[0].mean() + style_loss_maps[1].mean()
        loss_gan = self.gan_loss.compute_generator_loss(self.discriminator, generated_texture_l)
        patch_gt_l, patch_generated_l = TextureMapDataset.sample_patches(batch['mask_texture'], self.hparams.patch_size, self.hparams.num_patches, gt_texture_l, generated_texture_l)
        loss_gan_local = self.gan_loss.compute_generator_loss(self.discriminator_local, torch.cat([patch_gt_l, patch_generated_l], dim=1))
        loss_total = loss_regression_l * self.hparams.lambda_regr_l + loss_regression_ab * self.hparams.lambda_regr_ab + loss_content * self.hparams.lambda_content + loss_style * self.hparams.lambda_style \
                     + loss_gan * self.hparams.lambda_g + loss_gan_local * self.hparams.lambda_g_local
        return loss_total, loss_regression_l, loss_regression_ab, loss_content, loss_style, loss_gan, loss_gan_local

    def training_step(self, batch, batch_index, optimizer_idx):
        if optimizer_idx == 0:
            generated_texture = self.forward(batch)
            loss_total, loss_regression_l, loss_regression_ab, loss_content, loss_style, loss_gan, loss_gan_local = self.calculate_losses(generated_texture, batch)
            self.log('train/loss_regression_l', loss_regression_l, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log('train/loss_regression_ab', loss_regression_ab, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log('train/loss_content', loss_content, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log('train/loss_style', loss_style, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log('train/loss_total', loss_total, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'adversarial/G', loss_gan, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
            self.log(f'adversarial_local/G', loss_gan_local, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        else:
            loss_total = torch.zeros([1], dtype=batch['texture'].dtype, requires_grad=True)
            if int(self.hparams.lambda_g > 0):
                with torch.no_grad():
                    generated_texture = self.forward(batch)
                gt_texture_l, _ = self.split_into_channels(batch['texture'])
                generated_texture_l, _ = self.split_into_channels(generated_texture)
                if optimizer_idx == 1:
                    d_real_loss, d_fake_loss, d_gp_loss = self.gan_loss.compute_discriminator_loss(self.discriminator, gt_texture_l, generated_texture_l.detach())
                else:
                    patch_gt_l_0, patch_gt_l_1, patch_generated_l = TextureMapDataset.sample_patches(batch['mask_texture'], self.hparams.patch_size, self.hparams.num_patches, gt_texture_l, gt_texture_l, generated_texture_l)
                    d_real_loss, d_fake_loss, d_gp_loss = self.gan_loss.compute_discriminator_loss(self.discriminator_local, torch.cat([patch_gt_l_0, patch_gt_l_1], dim=1), torch.cat([patch_gt_l_0, patch_generated_l], dim=1).detach())
                loss_total = d_real_loss + d_fake_loss + self.hparams.lambda_gp * d_gp_loss
                suffix = ["", "_local"][optimizer_idx - 1]
                self.log(f'adversarial{suffix}/D_real', d_real_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
                self.log(f'adversarial{suffix}/D_fake', d_fake_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
                self.log(f'adversarial{suffix}/D_gp', d_gp_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
                self.log(f'adversarial{suffix}/D', d_real_loss + d_fake_loss, on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        return {'loss': loss_total}

    def validation_step(self, batch, batch_index, dataloader_index):
        split = ["val", "train"][dataloader_index]
        suffix = ["", "_epoch"][dataloader_index]
        generated_texture = self.forward(batch)
        loss_total, loss_regression_l, loss_regression_ab, loss_content, loss_style, _, _ = self.calculate_losses(generated_texture, batch)
        self.log(f'{split}/loss_regression_l{suffix}', loss_regression_l, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f'{split}/loss_regression_ab{suffix}', loss_regression_ab, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f'{split}/loss_content{suffix}', loss_content, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f'{split}/loss_style{suffix}', loss_style, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f'{split}/loss_total{suffix}', loss_total, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)

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
                    generated_texture = self.forward(batch)
                    gt_texture_l, gt_texture_ab = self.split_into_channels(batch['texture'])
                    generated_texture_l, generated_texture_ab = self.split_into_channels(generated_texture)
                    loss_regression_l = self.regression_loss.calculate_loss(gt_texture_l, generated_texture_l).mean(axis=1).squeeze(1)
                    loss_regression_ab = self.regression_loss.calculate_loss(gt_texture_ab, generated_texture_ab).mean(axis=1).squeeze(1)
                    loss_content = self.feature_loss_helper.calculate_feature_loss(gt_texture_l, generated_texture_l).mean(axis=1).squeeze(1)
                    style_loss_maps = self.feature_loss_helper.calculate_style_loss(gt_texture_l, generated_texture_l)
                    loss_style = (torch.nn.functional.interpolate(style_loss_maps[1].unsqueeze(1), size=style_loss_maps[0].shape[1:]).squeeze(1) + style_loss_maps[0]) / 2
                    for ii in range(generated_texture.shape[0]):
                        self.visualize_prediction(output_vis_path, batch['name'][ii], batch['view_index'][ii], batch['texture'][ii].cpu().numpy(), batch['render'][ii].cpu().numpy(), batch['partial_texture'][ii].cpu().numpy(),
                                                  generated_texture[ii].cpu().numpy(), loss_regression_l[ii].cpu().numpy(), loss_regression_ab[ii].cpu().numpy(), loss_content[ii].cpu().numpy(), loss_style[ii].cpu().numpy())

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        opt_d_local = torch.optim.Adam(self.discriminator_local.parameters(), lr=self.hparams.lr, betas=(0.5, 0.999))
        return [opt_g, opt_d, opt_d_local], []

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return [torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=False),
                torch.utils.data.DataLoader(self.train_val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=False)]

    def visualize_prediction(self, save_dir, name, v_idx, texture, render, partial_texture, prediction, loss_regression_l, loss_regression_ab, loss_content, loss_style):
        import matplotlib.pyplot as plt
        [texture, prediction, render, partial_texture], _, _ = self.train_dataset.convert_data_for_visualization([texture, prediction, render, partial_texture], [], [])
        f, axarr = plt.subplots(1, 9, figsize=(36, 4))
        items = [render, partial_texture, texture, prediction]
        for i in range(4):
            axarr[i].imshow(items[i])
            axarr[i].axis('off')
        items = [loss_regression_l, loss_regression_ab, loss_content, loss_style]
        for i in range(4):
            items[i] = (items[i] - items[i].min()) / (items[i].max() - items[i].min())
            axarr[4 + i].imshow(items[i], cmap='jet')
            axarr[4 + i].axis('off')
        closest_plotted = False
        closest_train = Path(self.hparams.dataset.data_dir) / 'splits' / self.hparams.dataset.name / 'closest_train.json'
        if closest_train.exists():
            closest_train_dict = json.loads(closest_train.read_text())
            if name in closest_train_dict:
                texture_path = self.train_dataset.path_to_dataset / closest_train_dict[name] / "surface_texture.png"
                if texture_path.exists():
                    with Image.open(texture_path) as texture_im:
                        closest = TextureMapDataset.process_to_padded_thumbnail(texture_im, self.train_dataset.texture_map_size) / 255
                    axarr[8].imshow(closest)
                    axarr[8].axis('off')
                    closest_plotted = True
        if not closest_plotted:
            axarr[8].imshow(np.zeros_like(loss_content), cmap='binary')
            axarr[8].axis('off')
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
        Image.fromarray((texture * 255).astype(np.uint8)).save(save_dir / "meshes" / f"{name}_gt.jpg")
        Image.fromarray((prediction * 255).astype(np.uint8)).save(save_dir / "meshes" / f"{name}_pred.jpg")

    def on_post_move_to_device(self):
        self.feature_loss_helper.move_to_device(self.device)


@hydra.main(config_path='../config', config_name='gan_conditional')
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
