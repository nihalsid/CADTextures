import shutil
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
import torch_scatter
from tabulate import tabulate
from cleanfid import fid
from pytorch_lightning.utilities import rank_zero_only
from torch_geometric.nn import GraphNorm, BatchNorm

from dataset.graph_mesh_dataset import GraphMeshDataset, FaceGraphMeshDataset, GraphDataLoader
from model.differentiable_renderer import DifferentiableRenderer
from model.graphnet import BigGraphSAGEEncoderDecoder, BigFaceEncoderDecoder, FaceConv, SymmetricFaceConv, SpatialAttentionConv, WrappedLinear
from util.create_trainer import create_trainer
from util.feature_loss import FeatureLossHelper
from util.misc import get_tensor_as_image
from util.regression_loss import RegressionLossHelper
from apex.optimizers import FusedAdam


class Supervise2DTrainer(pl.LightningModule):

    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.model = self.get_model()
        self.to_vertex_colors = to_vertex_colors_stub if config.method == 'graph' else to_vertex_colors_scatter
        DatasetCls = self.get_dataset_class()
        self.trainset = DatasetCls(config, 'train', use_single_view=config.dataset.single_view, load_to_memory=config.dataset.memory)
        self.valset = DatasetCls(config, 'val', use_single_view=config.dataset.single_view)
        self.trainvalset = DatasetCls(config, 'train_val', use_single_view=config.dataset.single_view)
        self.valvisset = DatasetCls(config, 'val_vis', use_single_view=True)
        self.render_helper = None
        self.l1_criterion = RegressionLossHelper('l1')
        self.l2_criterion = RegressionLossHelper('l2')
        self.feature_loss_helper = FeatureLossHelper(['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'],
                                                     [1 / 8, 1 / 4, 1 / 2, 1], [1 / 32, 1 / 16, 1 / 8, 1 / 4])

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)
        scheduler = []
        if self.config.scheduler is not None:
            scheduler = [torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.config.scheduler, gamma=0.5)]
        return [optimizer], scheduler

    def forward(self, batch, return_face_colors=False):
        pred_face_colors = self.model(batch["x"], batch["graph_data"])
        tgt_face_colors = batch["y"]
        rendered_color_pred = self.render_helper.render(batch['vertices'], batch['indices'], self.to_vertex_colors(pred_face_colors, batch), batch["ranges"].cpu())
        rendered_color_gt = self.render_helper.render(batch['vertices'], batch['indices'], self.to_vertex_colors(tgt_face_colors, batch), batch["ranges"].cpu())
        if return_face_colors:
            return pred_face_colors, rendered_color_pred.permute((0, 3, 1, 2)), rendered_color_gt.permute((0, 3, 1, 2))
        return rendered_color_pred.permute((0, 3, 1, 2)), rendered_color_gt.permute((0, 3, 1, 2))

    def training_step(self, batch, batch_idx):
        rendered_color_pred, rendered_color_gt = self.forward(batch)
        loss_l1 = self.l1_criterion.calculate_loss(rendered_color_gt, rendered_color_pred).mean()
        loss_content, loss_style = self.feature_loss_helper.calculate_perceptual_losses(rendered_color_gt, rendered_color_pred)
        loss_total = self.config.w_l1 * loss_l1 + self.config.w_content * loss_content + self.config.w_style * loss_style
        self.log("train_step/loss_l1", loss_l1, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, add_dataloader_idx=False)
        self.log("train_step/loss_total", loss_total, on_step=False, on_epoch=True, prog_bar=False, logger=True, sync_dist=True, add_dataloader_idx=False)
        return loss_total

    def validation_step(self, batch, batch_idx, dataloader_idx):
        phase = ["val", "train", "vis"][dataloader_idx]
        export_func = [self.handle_fid_export_texture, self.handle_fid_export_texture, self.handle_vis_export][dataloader_idx]
        rendered_color_in = self.render_helper.render(batch['vertices'],
                                                      batch['indices'],
                                                      self.to_vertex_colors(batch["x"][:, 3:6], batch).contiguous(),
                                                      batch["ranges"].cpu()).permute((0, 3, 1, 2))
        face_color_pred, rendered_color_pred, rendered_color_gt = self.forward(batch, return_face_colors=True)
        loss_l1_view = self.l1_criterion.calculate_loss(rendered_color_gt, rendered_color_pred).mean()
        loss_lpips_view = self.feature_loss_helper.model_alex(rendered_color_gt * 2, rendered_color_pred * 2).mean()
        loss_content, loss_style = self.feature_loss_helper.calculate_perceptual_losses(rendered_color_gt, rendered_color_pred)

        # for now calculate losses in texture space (so we can compare with 3d scenario
        # todo: eventually phase these out
        mask = self.valset.mask(batch["y"], self.config.batch_size).unsqueeze(-1)
        loss_l1_tex = (self.l1_criterion.calculate_loss(face_color_pred, batch["y"]).mean(dim=1) * mask).mean().item()
        prediction_as_image, target_as_image, input_as_image = self.get_pred_batch_as_texture(face_color_pred, batch)
        loss_lpips_tex = self.feature_loss_helper.model_alex(target_as_image * 2, prediction_as_image * 2).mean()

        self.log(f"{phase}/loss_l1", loss_l1_tex, on_step=False, on_epoch=True, prog_bar=False, logger=True, add_dataloader_idx=False)
        self.log(f"{phase}/loss_content", loss_content, on_step=False, on_epoch=True, prog_bar=False, logger=True, add_dataloader_idx=False)
        self.log(f"{phase}/loss_style", loss_style, on_step=False, on_epoch=True, prog_bar=False, logger=True, add_dataloader_idx=False)
        self.log(f"{phase}/loss_lpips", loss_lpips_tex, on_step=False, on_epoch=True, prog_bar=False, logger=True, add_dataloader_idx=False)
        export_func(face_color_pred, batch, rendered_color_in, rendered_color_gt, rendered_color_pred, batch['name'], phase)

    @rank_zero_only
    def validation_epoch_end(self, _val_step_outputs):
        for phase in ["val", "train"]:
            fid_score = fid.compute_fid(f'runs/{self.config.experiment}/fid/{phase}/real',
                                        f'runs/{self.config.experiment}/fid/{phase}/fake')
            kid_score = fid.compute_kid(f'runs/{self.config.experiment}/fid/{phase}/real',
                                        f'runs/{self.config.experiment}/fid/{phase}/fake')
            self.log(f"{phase}/loss_fid", fid_score, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True, add_dataloader_idx=False)
            self.log(f"{phase}/loss_kid", kid_score, on_step=False, on_epoch=True, prog_bar=False, logger=True, rank_zero_only=True, add_dataloader_idx=False)
        shutil.rmtree(f'runs/{self.config.experiment}/fid')
        cm = self.trainer.callback_metrics
        table_data = [["split", "l1", "lpips", "kid", "fid"],
                      ["train", cm["train/loss_l1"], cm["train/loss_lpips"], cm["train/loss_kid"], cm["train/loss_fid"]],
                      ["val", cm["val/loss_l1"], cm["val/loss_lpips"], cm["val/loss_kid"], cm["val/loss_fid"]]]
        print(tabulate(table_data, headers='firstrow', tablefmt='psql', floatfmt=".4f"))

    # ======= Utility Methods =======

    def train_dataloader(self):
        return GraphDataLoader(self.trainset, self.config.batch_size, shuffle=True, pin_memory=True, drop_last=True, num_workers=self.config.num_workers)

    def val_dataloader(self):
        return [
            GraphDataLoader(self.valset, self.config.batch_size, shuffle=False, num_workers=0),
            GraphDataLoader(self.trainvalset, self.config.batch_size, shuffle=False, num_workers=0),
            GraphDataLoader(self.valvisset, self.config.batch_size, shuffle=False, num_workers=0),
        ]

    def get_pred_batch_as_texture(self, face_color_pred, batch):
        mask = self.valset.mask(batch["y"], self.config.batch_size).unsqueeze(-1)
        input_as_image = self.valset.to_image((batch["x"][:, 3:6] * mask), batch["graph_data"]["level_masks"][0])
        prediction_as_image = self.valset.to_image((face_color_pred * mask), batch["graph_data"]["level_masks"][0])
        target_as_image = self.valset.to_image((batch["y"] * mask), batch["graph_data"]["level_masks"][0])
        return prediction_as_image, target_as_image, input_as_image

    def handle_vis_export(self, face_color_pred, batch, rendered_color_in, rendered_color_gt, rendered_color_pred, names, _phase):
        output_dir_view = Path(f'runs/{self.config.experiment}/visualization/epoch_{self.current_epoch:04d}/view')
        output_dir_tex = Path(f'runs/{self.config.experiment}/visualization/epoch_{self.current_epoch:04d}/texture')
        output_dir_view.mkdir(exist_ok=True, parents=True)
        output_dir_tex.mkdir(exist_ok=True, parents=True)
        prediction_as_image, target_as_image, input_as_image = self.get_pred_batch_as_texture(face_color_pred, batch)
        for bi in range(rendered_color_gt.shape[0]):
            get_tensor_as_image(rendered_color_in[bi]).save(output_dir_view / f'in_{names[bi // batch["mvp"].shape[1]]}_{bi % batch["mvp"].shape[1]}.png')
            get_tensor_as_image(rendered_color_gt[bi]).save(output_dir_view / f'tgt_{names[bi // batch["mvp"].shape[1]]}_{bi % batch["mvp"].shape[1]}.png')
            get_tensor_as_image(rendered_color_pred[bi]).save(output_dir_view / f'pred_{names[bi // batch["mvp"].shape[1]]}_{bi % batch["mvp"].shape[1]}.png')
        for bi in range(target_as_image.shape[0]):
            get_tensor_as_image(input_as_image[bi]).save(output_dir_tex / f'in_{names[bi]}.png')
            get_tensor_as_image(target_as_image[bi]).save(output_dir_tex / f'tgt_{names[bi]}.png')
            get_tensor_as_image(prediction_as_image[bi]).save(output_dir_tex / f'pred_{names[bi]}.png')

    def handle_fid_export_view(self, _face_colors, batch, _rendered_color_in, rendered_color_gt, rendered_color_pred, names, phase):
        # create images for fid & kid
        path_real = Path(f'runs/{self.config.experiment}/fid/{phase}/real')
        path_fake = Path(f'runs/{self.config.experiment}/fid/{phase}/fake')
        path_real.mkdir(exist_ok=True, parents=True)
        path_fake.mkdir(exist_ok=True, parents=True)
        for bi in range(rendered_color_gt.shape[0]):
            get_tensor_as_image(rendered_color_gt[bi]).save(path_real / f'tgt_{names[bi // batch["mvp"].shape[1]]}_{bi % batch["mvp"].shape[1]}.png')
            get_tensor_as_image(rendered_color_pred[bi]).save(path_fake / f'pred_{names[bi // batch["mvp"].shape[1]]}_{bi % batch["mvp"].shape[1]}.png')

    def handle_fid_export_texture(self, face_color_pred, batch, _rendered_color_in, _rendered_color_gt, _rendered_color_pred, names, phase):
        # create images for fid & kid
        path_real = Path(f'runs/{self.config.experiment}/fid/{phase}/real')
        path_fake = Path(f'runs/{self.config.experiment}/fid/{phase}/fake')
        path_real.mkdir(exist_ok=True, parents=True)
        path_fake.mkdir(exist_ok=True, parents=True)
        prediction_as_image, target_as_image, _ = self.get_pred_batch_as_texture(face_color_pred, batch)
        for bi in range(target_as_image.shape[0]):
            get_tensor_as_image(target_as_image[bi]).save(path_real / f'tgt_{names[bi]}.png')
            get_tensor_as_image(prediction_as_image[bi]).save(path_fake / f'pred_{names[bi]}.png')

    def get_dataset_class(self):
        return GraphMeshDataset if self.config.method == 'graph' else FaceGraphMeshDataset

    def get_model(self):
        model = None
        norm = BatchNorm if self.config.batch_size > 1 else GraphNorm
        if self.config.method == 'graph':
            input_feats = 3 + 3 + 1 + 6
            model = BigGraphSAGEEncoderDecoder(input_feats, 3, self.config.nf, 'max', num_pools=self.config.dataset.num_pools, norm=norm)
        else:
            input_feats = 3 + 3 + 1 + 3
            if self.config.conv == 'cartesian':
                conv_layer = lambda in_channels, out_channels: FaceConv(in_channels, out_channels, 8)
                model = BigFaceEncoderDecoder(input_feats, 3, self.config.nf, conv_layer, num_pools=self.config.dataset.num_pools,
                                              norm=norm, use_blur=self.config.use_blur, use_self_attn=self.config.use_self_attn)
            elif self.config.conv == 'symmetric':
                conv_layer = lambda in_channels, out_channels: SymmetricFaceConv(in_channels, out_channels)
                model = BigFaceEncoderDecoder(input_feats, 3, self.config.nf, conv_layer, num_pools=self.config.dataset.num_pools,
                                              norm=norm, use_blur=self.config.use_blur, use_self_attn=self.config.use_self_attn)
            elif self.config.conv == 'attention':
                conv_layer = lambda in_channels, out_channels: SpatialAttentionConv(in_channels, out_channels)
                model = BigFaceEncoderDecoder(input_feats, 3, self.config.nf, conv_layer, num_pools=self.config.dataset.num_pools, input_transform=WrappedLinear,
                                              norm=norm, use_blur=self.config.use_blur, use_self_attn=self.config.use_self_attn)
        return model

    def on_train_start(self):
        if self.render_helper is None:
            self.render_helper = DifferentiableRenderer(224)
        self.feature_loss_helper.move_to_device(self.device)
        self.feature_loss_helper.model_alex.to(self.device)

    def on_validation_start(self):
        if self.render_helper is None:
            self.render_helper = DifferentiableRenderer(224)
        self.feature_loss_helper.move_to_device(self.device)
        self.feature_loss_helper.model_alex.to(self.device)


def to_vertex_colors_scatter(face_colors, batch):
    vertex_colors = torch.zeros((batch["vertices"].shape[0] // batch["mvp"].shape[1], face_colors.shape[1])).to(face_colors.device)
    torch_scatter.scatter_mean(face_colors.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3), batch["indices_quad"].reshape(-1).long(), dim=0, out=vertex_colors)
    return vertex_colors[batch['vertex_ctr'], :]


def to_vertex_colors_stub(vertex_colors, batch):
    return vertex_colors[batch['vertex_ctr'], :]


@hydra.main(config_path='../config', config_name='supervise_2d')
def main(config):
    trainer = create_trainer("Supervise2D", config)
    model = Supervise2DTrainer(config)
    trainer.fit(model)


if __name__ == '__main__':
    main()
