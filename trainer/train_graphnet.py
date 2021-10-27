import random
from pathlib import Path
from random import randint

import hydra
import lpips
import numpy as np
import torch
import wandb
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm

from dataset.graph_mesh_dataset import GraphMeshDataset, FaceGraphMeshDataset
from model.augmentation import rgb_shift, channel_dropout, channel_shuffle, random_gamma, random_brightness_contrast
from model.graphnet import BigGraphSAGEEncoderDecoder, BigFaceEncoderDecoder, FaceConv, SymmetricFaceConv, SpatialAttentionConv, WrappedLinear
from util.feature_loss import FeatureLossHelper
from util.misc import print_model_parameter_count
from util.regression_loss import RegressionLossHelper

torch.backends.cudnn.benchmark = True


class GraphNetTrainer:

    def __init__(self, config, logger):
        # declare device
        self.device = torch.device('cuda:0')
        model = None

        # instantiate model
        if config.method == 'graph':
            trainset = GraphMeshDataset(config, 'train', use_single_view=config.dataset.single_view, load_to_memory=config.dataset.memory)
            valset = GraphMeshDataset(config, 'val', use_single_view=config.dataset.single_view, use_all_views=False)
            valvisset = GraphMeshDataset(config, 'val_vis', use_single_view=True)
            input_feats = 3 + 3 + 1
            if not config.dataset.plane:
                input_feats += 6
            model = BigGraphSAGEEncoderDecoder(input_feats, 3, 256, 'max', num_pools=config.dataset.num_pools)
        else:
            trainset = FaceGraphMeshDataset(config, 'train', use_single_view=config.dataset.single_view, load_to_memory=config.dataset.memory)
            valset = FaceGraphMeshDataset(config, 'val', use_single_view=config.dataset.single_view, use_all_views=False)
            valvisset = FaceGraphMeshDataset(config, 'val_vis', use_single_view=True)
            input_feats = 3 + 3 + 1
            if not config.dataset.plane:
                input_feats += 3
            if config.conv == 'cartesian':
                conv_layer = lambda in_channels, out_channels: FaceConv(in_channels, out_channels, 8)
                model = BigFaceEncoderDecoder(input_feats, 3, 128, conv_layer, num_pools=config.dataset.num_pools)
            elif config.conv == 'symmetric':
                conv_layer = lambda in_channels, out_channels: SymmetricFaceConv(in_channels, out_channels)
                model = BigFaceEncoderDecoder(input_feats, 3, 128, conv_layer, num_pools=config.dataset.num_pools)
            elif config.conv == 'attention':
                conv_layer = lambda in_channels, out_channels: SpatialAttentionConv(in_channels, out_channels)
                model = BigFaceEncoderDecoder(input_feats, 3, 128, conv_layer, num_pools=config.dataset.num_pools, input_transform=WrappedLinear)

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

        for epoch in range(self.config.max_epoch):

            shuffled_indices = list(range(len(self.trainset)))
            random.shuffle(shuffled_indices)
            pbar = tqdm(list(enumerate(shuffled_indices)), desc=f'Epoch {epoch:03d}')
            for iter_idx, i in pbar:
                sample = self.trainset[i]
                # move batch to device
                sample = sample.to(self.device)
                # zero out previously accumulated gradients
                optimizer.zero_grad()
                if self.config.use_augmentations:
                    if random.random() < 0.5:
                        choosen_augmentation = random.choice([rgb_shift, channel_dropout, channel_shuffle, random_gamma, random_brightness_contrast])
                        sample.x[:, 3:6], sample.y = choosen_augmentation(sample.x[:, 3:6], sample.y)
                        if random.random() < 0.1:
                            choosen_augmentation = random.choice([rgb_shift, channel_dropout, channel_shuffle, random_gamma, random_brightness_contrast])
                            sample.x[:, 3:6], sample.y = choosen_augmentation(sample.x[:, 3:6], sample.y)

                # forward pass
                prediction = self.model(sample.x, self.trainset.get_item_as_graphdata(sample))

                # compute loss
                mask = self.trainset.mask(sample.y)
                loss_l1 = (l1_criterion.calculate_loss(prediction[:sample.y.shape[0], :], sample.y).mean(dim=1) * mask).mean()

                prediction_as_image = self.trainset.to_image((prediction * mask.unsqueeze(-1))).unsqueeze(0)
                target_as_image = self.trainset.to_image((sample.y * mask.unsqueeze(-1))).unsqueeze(0)

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

                iteration = epoch * len(self.trainset) + iter_idx

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

                # set model to eval, important if your network has e.g. dropout or batchnorm layers
                self.model.eval()

                loss_total_val_l1 = 0
                loss_total_val_l2 = 0
                content_loss = 0
                style_loss = 0
                lpips_loss = 0

                # forward pass and evaluation for entire validation set
                for sample_val in self.valset:

                    sample_val = sample_val.to(self.device)

                    with torch.no_grad():
                        prediction = self.model(sample_val.x, self.valset.get_item_as_graphdata(sample_val))
                        prediction = prediction[:sample_val.y.shape[0], :]

                    mask = self.valset.mask(sample_val.y)
                    loss_total_val_l1 += (l1_criterion.calculate_loss(prediction, sample_val.y).mean(dim=1) * mask).mean().item()
                    loss_total_val_l2 += (l2_criterion.calculate_loss(prediction, sample_val.y).mean(dim=1) * mask).mean().item()
                    prediction_as_image = self.valset.to_image((prediction * mask.unsqueeze(-1))).unsqueeze(0)
                    target_as_image = self.valset.to_image((sample_val.y * mask.unsqueeze(-1))).unsqueeze(0)

                    # nihalsid: test image readoff from plane
                    # Path(f"runs/{config.experiment}/visualization/epoch_{epoch:05d}/").mkdir(exist_ok=True, parents=True)
                    # Image.fromarray(((prediction_as_image.squeeze(0).permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)).save(f'runs/{config.experiment}/visualization/epoch_{epoch:05d}/pred_{sample_val.name}.png')
                    # Image.fromarray(((target_as_image.squeeze(0).permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)).save(f'runs/{config.experiment}/visualization/epoch_{epoch:05d}/tgt_{sample_val.name}.png')

                    with torch.no_grad():
                        lpips_loss += loss_fn_alex(target_as_image * 2, prediction_as_image * 2).cpu().item()

                        loss_maps_content = feature_loss_helper.calculate_feature_loss(target_as_image, prediction_as_image)
                        content_loss = loss_maps_content[0].mean().item() * content_weights[0]
                        for loss_map_idx in range(1, len(loss_maps_content)):
                            content_loss += loss_maps_content[loss_map_idx].mean().item() * content_weights[loss_map_idx]

                        loss_maps_style = feature_loss_helper.calculate_style_loss(target_as_image, prediction_as_image)
                        style_loss = loss_maps_style[0].mean().item() * style_weights[0]
                        for loss_map_idx in range(1, len(loss_maps_style)):
                            style_loss += loss_maps_style[loss_map_idx].mean().item() * style_weights[loss_map_idx]

                for loss_name, loss_var in [("loss_val_l1", loss_total_val_l1), ("loss_val_l2", loss_total_val_l2),
                                            ("loss_val_style", style_loss * 1e3), ("loss_val_content", content_loss), ("lpips_loss", lpips_loss)]:
                    print(f'[{epoch:03d}] {loss_name}: {loss_var / len(self.valset):.5f}')
                    self.logger.log({loss_name: loss_var / len(self.valset), 'iter': epoch * len(self.trainset), 'epoch': epoch})

                if epoch % self.config.save_epoch == (self.config.save_epoch - 1):
                    torch.save(self.model.state_dict(), f'runs/{self.config.experiment}/model_{epoch}.ckpt')

                for sample_val in self.valvisset:
                    sample_val = sample_val.to(self.device)

                    with torch.no_grad():
                        prediction = self.model(sample_val.x, self.valset.get_item_as_graphdata(sample_val))
                        prediction = prediction[:sample_val.y.shape[0], :]

                    mask = self.valset.mask(sample_val.y).unsqueeze(-1)
                    self.valvisset.visualize_graph_with_predictions(sample_val, (sample_val.y * mask).cpu().numpy(), f'runs/{self.config.experiment}/visualization/epoch_{epoch:05d}', 'gt')
                    self.valvisset.visualize_graph_with_predictions(sample_val, (prediction * mask).cpu().numpy(), f'runs/{self.config.experiment}/visualization/epoch_{epoch:05d}', 'pred')
                    self.valvisset.visualize_graph_with_predictions(sample_val, (sample_val.x[:, 3:6] * mask).cpu().numpy(), f'runs/{self.config.experiment}/visualization/epoch_{epoch:05d}', 'in')

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
                    prediction = self.model(sample_val.x, sample_val.edge_index, sample_val.num_sub_vertices, sample_val.pool_maps, sample_val.sub_edges)
                else:
                    prediction = self.model(sample_val.x, sample_val.edge_index, sample_val.num_sub_vertices, sample_val.pool_maps, sample_val.pad_sizes, sample_val.sub_edges)
                prediction = prediction[:sample_val.y.shape[0], :]

            mask = self.valset.mask(sample_val.y)
            prediction_as_image = self.valset.to_image((prediction * mask.unsqueeze(-1))).unsqueeze(0)
            target_as_image = self.valset.to_image((sample_val.y * mask.unsqueeze(-1))).unsqueeze(0)

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
    ds_name = '_'.join(config.dataset.name.split('/'))
    logger = wandb.init(project=f'GATNet{config.suffix}[{ds_name}]', name=config.experiment, id=config.experiment)
    seed_everything(config.seed)
    trainer = GraphNetTrainer(config, logger)
    trainer.train()


if __name__ == '__main__':
    main()
