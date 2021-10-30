import random
import shutil
import signal
import time
from pathlib import Path
from random import randint

import hydra
import lpips
import numpy as np
import torch
import wandb
from PIL import Image
from cleanfid import fid
from pytorch_lightning import seed_everything
from torch_geometric.nn import BatchNorm, GraphNorm
from tqdm import tqdm

from dataset.graph_mesh_dataset import GraphMeshDataset, FaceGraphMeshDataset, to_device, GraphDataLoader
from model.augmentation import rgb_shift, channel_dropout, channel_shuffle, random_gamma, random_brightness_contrast
from model.graphnet import BigGraphSAGEEncoderDecoder, BigFaceEncoderDecoder, FaceConv, SymmetricFaceConv, SpatialAttentionConv, WrappedLinear
from util.feature_loss import FeatureLossHelper
from util.misc import print_model_parameter_count, register_debug_signal_handlers, register_quit_signal_handlers
from util.regression_loss import RegressionLossHelper

torch.backends.cudnn.benchmark = True


class GraphNetTrainer:

    def __init__(self, config, logger):
        # declare device
        self.device = torch.device('cuda:0')
        model = None
        if config.batch_size > 1:
            norm = BatchNorm
        else:
            norm = GraphNorm
        # instantiate model
        if config.method == 'graph':
            trainset = GraphMeshDataset(config, 'train', use_single_view=config.dataset.single_view, load_to_memory=config.dataset.memory)
            valset = GraphMeshDataset(config, 'val', use_single_view=config.dataset.single_view, use_all_views=False)
            trainvalset = GraphMeshDataset(config, 'train_val', use_single_view=config.dataset.single_view, use_all_views=False)
            valvisset = GraphMeshDataset(config, 'val_vis', use_single_view=True)
            input_feats = 3 + 3 + 1
            if not config.dataset.plane:
                input_feats += 6
            model = BigGraphSAGEEncoderDecoder(input_feats, 3, config.nf, 'max', num_pools=config.dataset.num_pools, norm=norm)
        else:
            trainset = FaceGraphMeshDataset(config, 'train', use_single_view=config.dataset.single_view, load_to_memory=config.dataset.memory)
            valset = FaceGraphMeshDataset(config, 'val', use_single_view=config.dataset.single_view, use_all_views=False)
            trainvalset = FaceGraphMeshDataset(config, 'train_val', use_single_view=config.dataset.single_view, use_all_views=False)
            valvisset = FaceGraphMeshDataset(config, 'val_vis', use_single_view=True)
            input_feats = 3 + 3 + 1
            if not config.dataset.plane:
                input_feats += 3
            if config.conv == 'cartesian':
                conv_layer = lambda in_channels, out_channels: FaceConv(in_channels, out_channels, 8)
                model = BigFaceEncoderDecoder(input_feats, 3, config.nf, conv_layer, num_pools=config.dataset.num_pools, norm=norm, use_blur=config.use_blur, use_self_attn=config.use_self_attn)
            elif config.conv == 'symmetric':
                conv_layer = lambda in_channels, out_channels: SymmetricFaceConv(in_channels, out_channels)
                model = BigFaceEncoderDecoder(input_feats, 3, config.nf, conv_layer, num_pools=config.dataset.num_pools, norm=norm, use_blur=config.use_blur, use_self_attn=config.use_self_attn)
            elif config.conv == 'attention':
                conv_layer = lambda in_channels, out_channels: SpatialAttentionConv(in_channels, out_channels)
                model = BigFaceEncoderDecoder(input_feats, 3, config.nf, conv_layer, num_pools=config.dataset.num_pools, input_transform=WrappedLinear, norm=norm, use_blur=config.use_blur, use_self_attn=config.use_self_attn)

        print(model)
        print_model_parameter_count(model)

        # load model if resuming from checkpoint
        if config.resume is not None:
            model.load_state_dict(torch.load(config.resume, map_location='cpu'))

        # move model to specified device
        model.to(self.device)

        self.model = model
        self.trainset = trainset
        self.valset = valset
        self.valvisset = valvisset
        self.trainvalset = trainvalset
        self.config = config
        self.logger = logger

    def train(self):
        # create folder for saving checkpoints
        Path(f'runs/{self.config.experiment}').mkdir(exist_ok=True, parents=True)

        # declare loss and move to specified device
        l1_criterion = RegressionLossHelper('l1')
        l2_criterion = RegressionLossHelper('l2')
        loss_fn_alex = lpips.LPIPS(net='alex').to(self.device)
        content_weights = [1 / 8, 1 / 4, 1 / 2, 1]
        style_weights = [1 / 32, 1 / 16, 1 / 8, 1 / 4]
        feature_loss_helper = FeatureLossHelper(['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], 'rgb')
        feature_loss_helper.move_to_device(self.device)

        # declare optimizer
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay=self.config.weight_decay)

        # set model to train, important if your network has e.g. dropout or batchnorm layers
        self.model.train()

        # keep track of running average of train loss for printing
        train_loss_running = 0.
        train_loss_running_total = 0.
        train_dataloader = GraphDataLoader(self.trainset, self.config.batch_size, shuffle=True, pin_memory=True, num_workers=self.config.num_workers)
        val_dataloader = GraphDataLoader(self.valset, self.config.batch_size, shuffle=False, num_workers=0)
        vis_dataloader = GraphDataLoader(self.valvisset, self.config.batch_size, shuffle=False, num_workers=0)
        trainval_dataloader = GraphDataLoader(self.trainvalset, self.config.batch_size, shuffle=False, num_workers=0)

        for epoch in range(self.config.max_epoch):
            pbar = tqdm(list(enumerate(train_dataloader)), desc=f'Epoch {epoch:03d}')
            for batch_idx, batch in pbar:
                sample = batch
                # move batch to device
                sample = to_device(sample, self.device)
                # zero out previously accumulated gradients
                optimizer.zero_grad(set_to_none=True)

                if self.config.use_augmentations:
                    if random.random() < 0.5:
                        choosen_augmentation = random.choice([rgb_shift, channel_dropout, channel_shuffle, random_gamma, random_brightness_contrast])
                        sample["x"][:, 3:6], sample["y"] = choosen_augmentation(sample["x"][:, 3:6], sample["y"])
                        if random.random() < 0.1:
                            choosen_augmentation = random.choice([rgb_shift, channel_dropout, channel_shuffle, random_gamma, random_brightness_contrast])
                            sample["x"][:, 3:6], sample["y"] = choosen_augmentation(sample["x"][:, 3:6], sample["y"])

                # forward pass
                prediction = self.model(sample["x"], sample["graph_data"])

                # compute loss
                mask = self.trainset.mask(sample["y"], self.config.batch_size)
                loss_l1 = (l1_criterion.calculate_loss(prediction[:sample["y"].shape[0], :], sample["y"]).mean(dim=1) * mask).mean()

                prediction_as_image = self.trainset.to_image((prediction * mask.unsqueeze(-1)), sample["graph_data"]["level_masks"][0])
                target_as_image = self.trainset.to_image((sample["y"] * mask.unsqueeze(-1)), sample["graph_data"]["level_masks"][0])

                loss_maps_content = feature_loss_helper.calculate_feature_loss(target_as_image, prediction_as_image)
                loss_content = loss_maps_content[0].mean() * content_weights[0]
                for loss_map_idx in range(1, len(loss_maps_content)):
                    loss_content += loss_maps_content[loss_map_idx].mean() * content_weights[loss_map_idx]

                loss_maps_style = feature_loss_helper.calculate_style_loss(target_as_image, prediction_as_image)
                loss_style = loss_maps_style[0].mean() * style_weights[0]
                for loss_map_idx in range(1, len(loss_maps_style)):
                    loss_style += loss_maps_style[loss_map_idx].mean() * style_weights[loss_map_idx]

                loss_total = self.config.w_l1 * loss_l1 + self.config.w_content * loss_content + self.config.w_style * loss_style

                # compute gradients on loss_total
                loss_total.backward()

                # update network params
                optimizer.step()

                # loss logging
                train_loss_running += loss_l1.item()
                train_loss_running_total += loss_total.item()

                iteration = epoch * len(train_dataloader) + batch_idx

                if iteration % self.config.print_interval == (self.config.print_interval - 1):
                    last_train_loss = train_loss_running / self.config.print_interval
                    last_train_loss_total = train_loss_running_total / self.config.print_interval
                    self.logger.log({'loss_train': last_train_loss, 'iter': iteration, 'epoch': epoch})
                    self.logger.log({'loss_train_total': last_train_loss_total, 'iter': iteration, 'epoch': epoch})
                    train_loss_running = 0.
                    train_loss_running_total = 0.
                    pbar.set_postfix({'loss': f'{last_train_loss_total:.4f}'})

            # validation evaluation and logging
            if epoch % self.config.val_check_interval == (self.config.val_check_interval - 1):
                val_t0 = time.time()
                # set model to eval, important if your network has e.g. dropout or batchnorm layers
                self.model.eval()

                for eval_name, eval_loader in [("val", val_dataloader), ("train", trainval_dataloader)]:

                    loss_total_eval_l1 = 0
                    loss_total_eval_l2 = 0
                    content_loss = 0
                    style_loss = 0
                    lpips_loss = 0

                    Path(f'runs/{self.config.experiment}/fid/real').mkdir(exist_ok=True, parents=True)
                    Path(f'runs/{self.config.experiment}/fid/fake').mkdir(exist_ok=True, parents=True)

                    # forward pass and evaluation for entire validation set
                    for sample_eval in eval_loader:

                        sample_eval = to_device(sample_eval, self.device)

                        with torch.no_grad():
                            prediction = self.model(sample_eval["x"], sample_eval["graph_data"])
                            prediction = prediction[:sample_eval["y"].shape[0], :]

                        mask = self.valset.mask(sample_eval["y"], self.config.batch_size)
                        loss_total_eval_l1 += (l1_criterion.calculate_loss(prediction, sample_eval["y"]).mean(dim=1) * mask).mean().item()
                        loss_total_eval_l2 += (l2_criterion.calculate_loss(prediction, sample_eval["y"]).mean(dim=1) * mask).mean().item()
                        prediction_as_image = self.valset.to_image((prediction * mask.unsqueeze(-1)), sample_eval["graph_data"]["level_masks"][0])
                        target_as_image = self.valset.to_image((sample_eval["y"] * mask.unsqueeze(-1)), sample_eval["graph_data"]["level_masks"][0])

                        for bi in range(sample_eval["graph_data"]["level_masks"][0].max() + 1):
                            Image.fromarray(((prediction_as_image[bi].permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)).save(f'runs/{self.config.experiment}/fid/fake/pred_{sample_eval["name"][bi]}.png')
                            Image.fromarray(((target_as_image[bi].permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)).save(f'runs/{self.config.experiment}/fid/real/tgt_{sample_eval["name"][bi]}.png')

                        with torch.no_grad():
                            lpips_loss += loss_fn_alex(target_as_image * 2, prediction_as_image * 2).mean().cpu().item()

                            loss_maps_content = feature_loss_helper.calculate_feature_loss(target_as_image, prediction_as_image)
                            content_loss = loss_maps_content[0].mean().item() * content_weights[0]
                            for loss_map_idx in range(1, len(loss_maps_content)):
                                content_loss += loss_maps_content[loss_map_idx].mean().item() * content_weights[loss_map_idx]

                            loss_maps_style = feature_loss_helper.calculate_style_loss(target_as_image, prediction_as_image)
                            style_loss = loss_maps_style[0].mean().item() * style_weights[0]
                            for loss_map_idx in range(1, len(loss_maps_style)):
                                style_loss += loss_maps_style[loss_map_idx].mean().item() * style_weights[loss_map_idx]

                    fid_score = fid.compute_fid(f'runs/{self.config.experiment}/fid/real', f'runs/{self.config.experiment}/fid/fake', mode="clean")
                    kid_score = fid.compute_kid(f'runs/{self.config.experiment}/fid/real', f'runs/{self.config.experiment}/fid/fake', mode="clean")
                    shutil.rmtree(f'runs/{self.config.experiment}/fid')

                    for loss_name, loss_var in [(f"loss_{eval_name}_l1", loss_total_eval_l1), (f"loss_{eval_name}_l2", loss_total_eval_l2),
                                                (f"loss_{eval_name}_style", style_loss * 1e3), (f"loss_{eval_name}_content", content_loss), (f"lpips_{eval_name}", lpips_loss)]:
                        print(f'[{epoch:03d}] {loss_name}: {loss_var / len(eval_loader):.5f}')
                        self.logger.log({loss_name: loss_var / len(eval_loader), 'iter': epoch * len(train_dataloader), 'epoch': epoch})

                    print(f'[{epoch:03d}] fid_{eval_name}: {fid_score:.3f}')
                    print(f'[{epoch:03d}] kid_{eval_name}: {kid_score:.5f}')
                    self.logger.log({f"fid_{eval_name}": fid_score, 'iter': epoch * len(train_dataloader), 'epoch': epoch})
                    self.logger.log({f"kid_{eval_name}": kid_score, 'iter': epoch * len(train_dataloader), 'epoch': epoch})

                if epoch % self.config.save_epoch == (self.config.save_epoch - 1):
                    torch.save(self.model.state_dict(), f'runs/{self.config.experiment}/model_{epoch}.ckpt')

                for sample_val in vis_dataloader:
                    sample_val = to_device(sample_val, self.device)

                    with torch.no_grad():
                        prediction = self.model(sample_val["x"], sample_val["graph_data"])
                        prediction = prediction[:sample_val["y"].shape[0], :]

                    mask = self.valset.mask(sample_val["y"], self.config.batch_size).unsqueeze(-1)
                    input_as_image = self.valset.to_image((sample_val["x"][:, 3:6] * mask), sample_val["graph_data"]["level_masks"][0])
                    prediction_as_image = self.valset.to_image((prediction * mask), sample_val["graph_data"]["level_masks"][0])
                    target_as_image = self.valset.to_image((sample_val["y"] * mask), sample_val["graph_data"]["level_masks"][0])

                    Path(f"runs/{self.config.experiment}/visualization/epoch_{epoch:05d}/").mkdir(exist_ok=True, parents=True)
                    for bi in range(sample_val["graph_data"]["level_masks"][0].max() + 1):
                        Image.fromarray(((input_as_image[bi].permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)).save(f'runs/{self.config.experiment}/visualization/epoch_{epoch:05d}/in_{sample_val["name"][bi]}.png')
                        Image.fromarray(((prediction_as_image[bi].permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)).save(f'runs/{self.config.experiment}/visualization/epoch_{epoch:05d}/pred_{sample_val["name"][bi]}.png')
                        Image.fromarray(((target_as_image[bi].permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)).save(f'runs/{self.config.experiment}/visualization/epoch_{epoch:05d}/tgt_{sample_val["name"][bi]}.png')
                        yi_masked = self.valset.batch_mask((sample_val["y"] * mask), sample_val['graph_data'], bi)
                        pred_masked = self.valset.batch_mask((prediction * mask), sample_val['graph_data'], bi)
                        input_masked = self.valset.batch_mask((sample_val["x"][:, 3:6] * mask), sample_val['graph_data'], bi)
                        self.valvisset.visualize_graph_with_predictions(sample_val['name'][bi], yi_masked.cpu().numpy(), f'runs/{self.config.experiment}/visualization/epoch_{epoch:05d}', 'gt')
                        self.valvisset.visualize_graph_with_predictions(sample_val['name'][bi], pred_masked.cpu().numpy(), f'runs/{self.config.experiment}/visualization/epoch_{epoch:05d}', 'pred')
                        self.valvisset.visualize_graph_with_predictions(sample_val['name'][bi], input_masked.cpu().numpy(), f'runs/{self.config.experiment}/visualization/epoch_{epoch:05d}', 'in')

                print(f"Time take for val & vis: {time.time() - val_t0:.3f}")
                # set model back to train
                self.model.train()

    def inference(self):
        if self.config.method == 'graph':
            valset = GraphMeshDataset(self.config, 'val', use_single_view=self.config.dataset.single_view, use_all_views=True)
        else:
            valset = FaceGraphMeshDataset(self.config, 'val', use_single_view=self.config.dataset.single_view, use_all_views=True)

        vis_dir = f"{Path(self.config.resume).parent}/all_vis_{Path(self.config.resume).name}/"
        Path(vis_dir).mkdir(exist_ok=True, parents=True)
        for sample_val in tqdm(valset):
            sample_val = sample_val.to(self.device)
            with torch.no_grad():
                if self.config.method == 'graph':
                    prediction = self.model(sample_val["x"], sample_val.edge_index, sample_val.num_sub_vertices, sample_val.pool_maps, sample_val.sub_edges)
                else:
                    prediction = self.model(sample_val["x"], sample_val.edge_index, sample_val.num_sub_vertices, sample_val.pool_maps, sample_val.pad_sizes, sample_val.sub_edges)
                prediction = prediction[:sample_val["y"].shape[0], :]

            mask = self.valset.mask(sample_val["y"])
            prediction_as_image = self.valset.to_image((prediction * mask.unsqueeze(-1))).unsqueeze(0)
            target_as_image = self.valset.to_image((sample_val["y"] * mask.unsqueeze(-1))).unsqueeze(0)

            Image.fromarray(((prediction_as_image.squeeze(0).permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)).save(f'{vis_dir}/pred_{sample_val.name}.png')
            Image.fromarray(((target_as_image.squeeze(0).permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)).save(f'{vis_dir}/tgt_{sample_val.name}.png')


@hydra.main(config_path='../config', config_name='graph_nn')
def main(config):
    from datetime import datetime
    if not config.wandb_main and config.suffix == '':
        config.suffix = '-dev'
    config.experiment = f"{datetime.now().strftime('%d%m%H%M')}_graphnn_{config['experiment']}"
    if config.val_check_interval > 1:
        config.val_check_interval = int(config.val_check_interval)
    if config.seed is None:
        config.seed = randint(0, 999)
    # config.lr = config.batch_size * config.lr
    ds_name = '_'.join(config.dataset.name.split('/'))
    logger = wandb.init(project=f'GATNet{config.suffix}[{ds_name}]', name=config.experiment, id=config.experiment)
    seed_everything(config.seed)

    register_debug_signal_handlers()
    register_quit_signal_handlers()

    trainer = GraphNetTrainer(config, logger)
    trainer.train()


if __name__ == '__main__':
    main()
