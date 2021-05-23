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
from model.refinement import get_model
from util.contrastive_loss import NTXentLoss
from util.feature_loss import FeatureLossHelper
from util.regression_loss import RegressionLossHelper


class TextureMapRefinementModule(pl.LightningModule):

    def __init__(self, config):
        super(TextureMapRefinementModule, self).__init__()
        self.save_hyperparameters(config)
        self.K = config.dictionary.K
        self.preload_dict = {}
        dataset = lambda split: TextureMapDataset(config, split, self.preload_dict)
        self.train_dataset, self.val_dataset, self.train_val_dataset, self.train_vis_dataset, self.val_vis_dataset = dataset('train'), dataset('val'), dataset('train_val'), dataset('train_vis'), dataset('val_vis')
        self.model = get_model(config)
        self.nt_xent_loss = NTXentLoss(float(config.temperature), 8, True)
        self.regression_loss = RegressionLossHelper(self.hparams.regression_loss_type)
        self.feature_loss_helper = FeatureLossHelper(['relu4_2'], ['relu3_2', 'relu4_2'])
        phase_func_list = [(self.optimizer_phase_0, self.training_step_phase_0), (self.optimizer_phase_1, self.training_step_phase_1), (self.optimizer_phase_2, self.training_step_phase_2), (self.optimizer_phase_3, self.training_step_phase_3)]
        self.init_opt_func = phase_func_list[config.current_phase][0]
        self.training_step = phase_func_list[config.current_phase][1]

    def forward_input(self, batch):
        input_maps = batch['mask_texture']
        input_maps = torch.cat([input_maps, batch['partial_texture']], dim=1)
        generated_texture = self.model.forward_input(input_maps)
        return TextureMapDataset.apply_mask_texture(generated_texture, batch['mask_texture'])

    def forward_texture(self, batch):
        generated_texture = self.model.forward_retrievals(batch['texture'])
        return TextureMapDataset.apply_mask_texture(generated_texture, batch['mask_texture'])

    def forward_features(self, batch):
        input_maps = batch['mask_texture']
        input_maps = torch.cat([input_maps, batch['partial_texture']], dim=1)
        features = self.model.forward_features(input_maps, batch['texture'])
        return features

    def forward_complete(self, batch):
        input_maps = batch['mask_texture']
        input_maps = torch.cat([input_maps, batch['partial_texture']], dim=1)
        generated_texture = self.model(input_maps, batch['retrievals'])
        features = self.model.forward_features(input_maps, batch['texture'])
        return (TextureMapDataset.apply_mask_texture(generated_texture, batch['mask_texture']), *features)

    def forward_attention_vis(self, batch):
        input_maps = batch['mask_texture']
        input_maps = torch.cat([input_maps, batch['partial_texture']], dim=1)
        generated_texture, attention_maps, score_maps = self.model.forward_debug(input_maps, batch['retrievals'])
        return TextureMapDataset.apply_mask_texture(generated_texture, batch['mask_texture']), attention_maps, score_maps

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
        loss_total = loss_regression_l * self.hparams.lambda_regr_l + loss_regression_ab * self.hparams.lambda_regr_ab + loss_content * self.hparams.lambda_content + loss_style * self.hparams.lambda_style
        return loss_total, loss_regression_l, loss_regression_ab, loss_content, loss_style

    def training_step_phase_0(self, batch, batch_idx):
        self.train_dataset.apply_batch_transforms(batch)
        generated_texture = self.forward_input(batch)
        loss_texgan, loss_regression_l, loss_regression_ab, loss_content, loss_style = self.calculate_losses(generated_texture, batch)
        self.log('train/loss_regression_l', loss_regression_l.detach().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/loss_regression_ab', loss_regression_ab.detach().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/loss_content', loss_content.detach().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/loss_style', loss_style.detach().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/loss_total', loss_texgan.detach().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        return {'loss': loss_texgan}

    def training_step_phase_1(self, batch, batch_idx):
        self.model.decoder.eval()
        self.train_dataset.apply_batch_transforms(batch)
        generated_texture = self.forward_texture(batch)
        loss_retrieval = self.regression_loss.calculate_loss(batch['texture'], generated_texture).mean()
        self.log('train/loss_retrieval', loss_retrieval.detach().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.model.decoder.train()
        return {'loss': loss_retrieval}

    def training_step_phase_2(self, batch, batch_idx):
        self.model.input_feature_extractor.eval()
        self.model.decoder.eval()
        self.train_dataset.apply_batch_transforms(batch)
        feat_x, feat_y = self.forward_features(batch)
        loss_ntxent = self.nt_xent_loss(feat_x, feat_y)
        self.log('train/loss_ntxent', loss_ntxent.detach().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.model.input_feature_extractor.train()
        self.model.decoder.train()
        return {'loss': loss_ntxent}

    def training_step_phase_3(self, batch, batch_idx):
        self.train_dataset.apply_batch_transforms(batch)
        generated_texture, feat_x, feat_y = self.forward_complete(batch)
        generated_tex_retrieval = self.forward_texture(batch)
        loss_texgan, loss_regression_l, loss_regression_ab, loss_content, loss_style = self.calculate_losses(generated_texture, batch)
        loss_ntxent = self.nt_xent_loss(feat_x, feat_y)
        loss_retrieval = self.regression_loss.calculate_loss(batch['texture'], generated_tex_retrieval).mean()
        loss_total = loss_texgan + self.hparams.lambda_regr_retr * loss_retrieval + self.hparams.lambda_ntxent * loss_ntxent
        self.log('train/loss_regression_l', loss_regression_l.detach().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/loss_regression_ab', loss_regression_ab.detach().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/loss_content', loss_content.detach().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/loss_style', loss_style.detach().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/loss_ntxent', loss_ntxent.detach().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/loss_retrieval', loss_retrieval.detach().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        self.log('train/loss_total', loss_total.detach().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True, sync_dist=True)
        return {'loss': loss_total}

    def validation_step(self, batch, batch_index, dataloader_index):
        self.val_dataset.apply_batch_transforms(batch)
        split = ["val", "train"][dataloader_index]
        suffix = ["", "_epoch"][dataloader_index]
        generated_texture, feat_x, feat_y = self.forward_complete(batch)
        loss_total, loss_regression_l, loss_regression_ab, loss_content, loss_style = self.calculate_losses(generated_texture, batch)
        loss_ntxent = self.nt_xent_loss(feat_x, feat_y)
        self.log(f'{split}/loss_regression_l{suffix}', loss_regression_l, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f'{split}/loss_regression_ab{suffix}', loss_regression_ab, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
        self.log(f'{split}/loss_ntxent{suffix}', loss_ntxent, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
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
                    ds.apply_batch_transforms(batch)
                    generated_texture, attention_maps, score_maps = self.forward_attention_vis(batch)
                    generated_bypassed_texture = self.forward_input(batch)
                    retrieval_reconstructed_0 = self.forward_texture(batch)
                    gt_texture_l, gt_texture_ab = self.split_into_channels(batch['texture'])
                    generated_texture_l, generated_texture_ab = self.split_into_channels(generated_texture)
                    loss_regression_l = self.regression_loss.calculate_loss(gt_texture_l, generated_texture_l).mean(axis=1).squeeze(1)
                    loss_regression_ab = self.regression_loss.calculate_loss(gt_texture_ab, generated_texture_ab).mean(axis=1).squeeze(1)
                    loss_content = self.feature_loss_helper.calculate_feature_loss(gt_texture_l, generated_texture_l).mean(axis=1).squeeze(1)
                    style_loss_maps = self.feature_loss_helper.calculate_style_loss(gt_texture_l, generated_texture_l)
                    loss_style = (torch.nn.functional.interpolate(style_loss_maps[1].unsqueeze(1), size=style_loss_maps[0].shape[1:]).squeeze(1) + style_loss_maps[0]) / 2
                    for ii in range(generated_texture.shape[0]):
                        self.visualize_prediction(output_vis_path, batch['name'][ii], batch['view_index'][ii], batch['texture'][ii].cpu().numpy(), batch['render'][ii].cpu().numpy(), batch['partial_texture'][ii].cpu().numpy(),
                                                  generated_texture[ii].cpu().numpy(), generated_bypassed_texture[ii].cpu().numpy(), batch['retrievals'][ii].cpu().numpy(), retrieval_reconstructed_0[ii].cpu().numpy(), attention_maps[ii].cpu().numpy(), score_maps[ii].cpu().numpy(),
                                                  loss_regression_l[ii].cpu().numpy(), loss_regression_ab[ii].cpu().numpy(), loss_content[ii].cpu().numpy(), loss_style[ii].cpu().numpy())

    def optimizer_phase_0(self):
        optimizer = torch.optim.Adam(list(self.model.input_feature_extractor.parameters()) + list(self.model.decoder.parameters()), lr=self.hparams["lr"])
        return [optimizer]

    def optimizer_phase_1(self):
        optimizer = torch.optim.Adam(list(self.model.retrieval_feature_extractor.parameters()), lr=self.hparams["lr"])
        return [optimizer]

    def optimizer_phase_2(self):
        optimizer = torch.optim.Adam(list(self.model.patch_attention.parameters()), lr=self.hparams["lr"])
        return [optimizer]

    def optimizer_phase_3(self):
        optimizer = torch.optim.Adam(list(self.model.parameters()), lr=self.hparams["lr"])
        return [optimizer]

    def configure_optimizers(self):
        return self.init_opt_func()

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.train_dataset, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=True)

    def val_dataloader(self):
        return [torch.utils.data.DataLoader(self.val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=False),
                torch.utils.data.DataLoader(self.train_val_dataset, batch_size=self.hparams.batch_size, shuffle=False, num_workers=self.hparams.num_workers, pin_memory=True, drop_last=False)]

    def visualize_prediction(self, save_dir, name, v_idx, texture, render, partial_texture, prediction, bypassed_prediction, retrievals, retrieval_reconstruction_0, attention_maps, score_maps, loss_regression_l, loss_regression_ab, loss_content, loss_style):
        import matplotlib.pyplot as plt
        [texture, prediction, bypassed_prediction, render, partial_texture, retrievals_0, retrieval_reconstruction_0], _, _ = self.train_dataset.convert_data_for_visualization([texture, prediction, bypassed_prediction, render, partial_texture, retrievals[0].copy(), retrieval_reconstruction_0], [], [])
        f, axarr = plt.subplots(1, 12, figsize=(48, 4))
        items = [render, partial_texture, texture, prediction, bypassed_prediction, retrievals_0, retrieval_reconstruction_0]
        for i in range(7):
            axarr[i].imshow(items[i])
            axarr[i].axis('off')
        items = [loss_regression_l, loss_regression_ab, loss_content, loss_style]
        for i in range(4):
            items[i] = (items[i] - items[i].min()) / (items[i].max() - items[i].min())
            axarr[7 + i].imshow(1 - items[i], cmap='RdYlGn')
            axarr[7 + i].axis('off')
        closest_plotted = False
        closest_train = Path(self.hparams.dataset.data_dir) / 'splits' / self.hparams.dataset.name / 'closest_train.json'
        if closest_train.exists():
            closest_train_dict = json.loads(closest_train.read_text())
            if name in closest_train_dict:
                texture_path = self.train_dataset.path_to_dataset / closest_train_dict[name] / "surface_texture.png"
                if texture_path.exists():
                    with Image.open(texture_path) as texture_im:
                        closest = TextureMapDataset.process_to_padded_thumbnail(texture_im, self.train_dataset.texture_map_size) / 255
                    axarr[11].imshow(closest)
                    axarr[11].axis('off')
                    closest_plotted = True
        if not closest_plotted:
            axarr[11].imshow(np.zeros_like(loss_content), cmap='binary')
            axarr[11].axis('off')
        plt.savefig(save_dir / "figures" / f"{name}_{v_idx}.jpg", bbox_inches='tight', dpi=360)
        plt.close()

        f, axarr = plt.subplots(3, self.K + 1, figsize=((self.K + 1) * 4, 8))
        retrievals_vis, _, _ = self.train_dataset.convert_data_for_visualization([retrievals[i] for i in range(self.K)], [], [])
        axarr[0, 0].imshow(texture)
        axarr[0, 0].axis('off')
        axarr[1, 0].imshow(prediction)
        axarr[1, 0].axis('off')
        axarr[2, 0].imshow(np.zeros_like(prediction))
        axarr[2, 0].axis('off')
        for i in range(self.K):
            axarr[0, 1 + i].imshow(retrievals_vis[i])
            axarr[0, 1 + i].axis('off')
            axarr[1, 1 + i].imshow(attention_maps[i], cmap='RdYlGn')
            axarr[1, 1 + i].axis('off')
            axarr[2, 1 + i].imshow((1 + score_maps[i]) / 2, cmap='RdYlGn')
            axarr[2, 1 + i].axis('off')
        plt.savefig(save_dir / "figures" / f"{name}_{v_idx}_attn.jpg", bbox_inches='tight', dpi=360)
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

    def on_load_checkpoint(self, checkpoint):
        checkpoint['optimizer_states'] = self.optimizers().state


@hydra.main(config_path='../config', config_name='refinement')
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

    max_loops = len(config.phase_change_epochs) - config.current_phase
    max_epochs = config.phase_change_epochs + [config.max_epoch]
    for i in range(len(max_epochs) - 1):
        max_epochs[i + 1] = max_epochs[i] + max_epochs[i + 1]

    print('Max loops: ', max_loops)
    print('Max epochs: ', max_epochs)
    print('Starting phase', config.current_phase)

    trainer = Trainer(gpus=[0], num_sanity_val_steps=config.sanity_steps, max_epochs=max_epochs[config.current_phase], limit_val_batches=config.val_check_percent, callbacks=[checkpoint_callback],
                      val_check_interval=float(min(config.val_check_interval, 1)), check_val_every_n_epoch=max(1, config.val_check_interval), resume_from_checkpoint=config.resume, logger=logger, benchmark=True)

    model = TextureMapRefinementModule(config)
    trainer.fit(model)

    for phase_idx in range(max_loops):
        config.current_phase += 1
        last_phase_ckpt = max([str(x) for x in (Path("runs") / config.experiment).iterdir() if x.name.endswith('.ckpt')], key=os.path.getctime)
        print('Starting phase', config.current_phase, '[' + last_phase_ckpt + ']', max_epochs[config.current_phase])
        model = TextureMapRefinementModule(config)
        trainer = Trainer(gpus=[0], num_sanity_val_steps=0, max_epochs=max_epochs[config.current_phase], limit_val_batches=config.val_check_percent, callbacks=[checkpoint_callback],
                          val_check_interval=float(min(config.val_check_interval, 1)), check_val_every_n_epoch=max(1, config.val_check_interval), resume_from_checkpoint=last_phase_ckpt, benchmark=True)
        trainer.fit(model)


if __name__ == '__main__':
    main()
