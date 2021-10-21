import random
from pathlib import Path
from random import randint

import hydra
import lpips
import wandb
import numpy as np
from PIL import Image
from pytorch_lightning import seed_everything
from tqdm import tqdm

from dataset.graph_mesh_dataset import GraphMeshDataset, FaceGraphMeshDataset
import torch
from model.graphnet import GATNet, GraphSAGENet, GCNNet, GraphUNet, GraphSAGEEncoderDecoder, BigGraphSAGEEncoderDecoder, BigFaceEncoderDecoder, FaceConv, SymmetricFaceConv, SpatialAttentionConv, WrappedLinear
from util.feature_loss import FeatureLossHelper
from util.misc import print_model_parameter_count
from util.regression_loss import RegressionLossHelper


def GraphNetTrainer(config, logger):

    # declare device
    device = torch.device('cuda:0')

    # instantiate model
    # model = GATNet(63 + 3 + 1 + 6, 3, 256, 0)
    # model = GraphSAGENet(3 + 3 + 1 + 6, 3, 256, 0)
    # model = GraphSAGEEncoderDecoder(3 + 3 + 1 + 6, 3, 64)
    # model = BigGraphSAGEEncoderDecoder(3 + 3 + 1, 3, 256, 'max')
    conv_layer = lambda in_channels, out_channels: FaceConv(in_channels, out_channels, 8)
    # conv_layer = lambda in_channels, out_channels: SymmetricFaceConv(in_channels, out_channels)
    # conv_layer = lambda in_channels, out_channels: SpatialAttentionConv(in_channels, out_channels)
    model = BigFaceEncoderDecoder(3 + 3 + 1, 3, 128, conv_layer)
    # model = BigFaceEncoderDecoder(3 + 3 + 1, 3, 128, conv_layer, WrappedLinear)
    # model = GCNNet(63 + 3 + 1 + 6, 3, 256 , 0)
    # wandb.watch(model, log='all')

    print_model_parameter_count(model)

    # create dataloaders
    trainset = FaceGraphMeshDataset(config, 'train', use_single_view=config.dataset.single_view, load_to_memory=config.dataset.memory)
    # trainset = GraphMeshDataset(config, 'train', use_single_view=config.dataset.single_view, load_to_memory=config.dataset.memory)

    valset = FaceGraphMeshDataset(config, 'val', use_single_view=config.dataset.single_view, use_all_views=False)
    # valset = GraphMeshDataset(config, 'val', use_single_view=config.dataset.single_view, use_all_views=False)

    valvisset = FaceGraphMeshDataset(config, 'val_vis', use_single_view=True)
    # valvisset = GraphMeshDataset(config, 'val_vis', use_single_view=True)

    # load model if resuming from checkpoint
    if config.resume is not None:
        model.load_state_dict(torch.load(config.resume, map_location='cpu'))

    # move model to specified device
    model.to(device)

    # create folder for saving checkpoints
    Path(f'runs/{config.experiment}').mkdir(exist_ok=True, parents=True)

    # start training
    train(model, trainset, valset, valvisset, device, config, logger)


def train(model, traindataset, valdataset, valvisdataset, device, config, logger):

    # declare loss and move to specified device
    loss_criterion = RegressionLossHelper('l1')
    l1_criterion = RegressionLossHelper('l1')
    l2_criterion = RegressionLossHelper('l2')
    loss_fn_alex = lpips.LPIPS(net='alex').to(device)
    feature_loss_helper = FeatureLossHelper(['relu4_2'], ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'], 'rgb')
    feature_loss_helper.move_to_device(device)

    # declare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # set model to train, important if your network has e.g. dropout or batchnorm layers
    model.train()

    # keep track of running average of train loss for printing
    train_loss_running = 0.

    for epoch in range(config.max_epoch):

        shuffled_indices = list(range(len(traindataset)))
        random.shuffle(shuffled_indices)
        pbar = tqdm(list(enumerate(shuffled_indices)), desc=f'Epoch {epoch:03d}')
        for iter_idx, i in pbar:
            sample = traindataset[i]
            # move batch to device
            sample = sample.to(device)
            # zero out previously accumulated gradients
            optimizer.zero_grad()

            # forward pass
            # prediction = model(sample.x, sample.edge_index, sample.num_sub_vertices, sample.pool_maps, sample.sub_edges)
            prediction = model(sample.x, sample.edge_index, sample.num_sub_vertices, sample.pool_maps, sample.pad_sizes, sample.sub_edges)

            # compute loss
            loss_total = (loss_criterion.calculate_loss(prediction[:sample.y.shape[0], :], sample.y).mean(dim=1) * traindataset.mask(sample.y)).mean()

            # compute gradients on loss_total
            loss_total.backward()

            # update network params
            optimizer.step()

            # loss logging
            train_loss_running += loss_total.item()
            iteration = epoch * len(traindataset) + iter_idx

            if iteration % config.print_interval == (config.print_interval - 1):
                last_train_loss = train_loss_running / config.print_interval
                logger.log({'loss_train': last_train_loss, 'iter': iteration, 'epoch': epoch})
                train_loss_running = 0.
                pbar.set_postfix({'loss': f'{last_train_loss:.4f}'})

        # validation evaluation and logging
        if epoch % config.val_check_interval == (config.val_check_interval - 1):

            # set model to eval, important if your network has e.g. dropout or batchnorm layers
            model.eval()

            loss_total_val_l1 = 0
            loss_total_val_l2 = 0
            content_loss = 0
            style_loss = 0
            lpips_loss = 0

            # forward pass and evaluation for entire validation set
            for sample_val in valdataset:

                sample_val = sample_val.to(device)

                with torch.no_grad():
                    # prediction = model(sample_val.x, sample_val.edge_index, sample_val.num_sub_vertices, sample_val.pool_maps, sample_val.sub_edges)
                    prediction = model(sample_val.x, sample_val.edge_index, sample_val.num_sub_vertices, sample_val.pool_maps, sample_val.pad_sizes, sample_val.sub_edges)
                    prediction = prediction[:sample_val.y.shape[0], :]

                mask = valdataset.mask(sample_val.y)
                loss_total_val_l1 += (l1_criterion.calculate_loss(prediction, sample_val.y).mean(dim=1) * mask).mean().item()
                loss_total_val_l2 += (l2_criterion.calculate_loss(prediction, sample_val.y).mean(dim=1) * mask).mean().item()
                prediction_as_image = valdataset.plane_to_image((prediction * mask.unsqueeze(-1)).cpu().numpy()).unsqueeze(0).to(prediction.device)
                target_as_image = valdataset.plane_to_image((sample_val.y * mask.unsqueeze(-1)).cpu().numpy()).unsqueeze(0).to(prediction.device)

                # nihalsid: test image readoff from plane
                # Path(f"runs/{config.experiment}/visualization/epoch_{epoch:05d}/").mkdir(exist_ok=True, parents=True)
                # Image.fromarray(((prediction_as_image.squeeze(0).permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)).save(f'runs/{config.experiment}/visualization/epoch_{epoch:05d}/pred_{sample_val.name}.jpg')
                # Image.fromarray(((target_as_image.squeeze(0).permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)).save(f'runs/{config.experiment}/visualization/epoch_{epoch:05d}/tgt_{sample_val.name}.jpg')

                with torch.no_grad():
                    lpips_loss += loss_fn_alex(target_as_image * 2, prediction_as_image * 2).cpu().item()
                    content_loss += feature_loss_helper.calculate_feature_loss(target_as_image, prediction_as_image).mean().item()
                    style_loss_maps = feature_loss_helper.calculate_style_loss(target_as_image, prediction_as_image)
                    style_loss += np.mean([style_loss_maps[map_idx].mean().item() for map_idx in range(len(style_loss_maps))])

            for loss_name, loss_var in [("loss_val_l1", loss_total_val_l1), ("loss_val_l2", loss_total_val_l2),
                                        ("loss_val_style", style_loss), ("loss_val_content", content_loss), ("lpips_loss", lpips_loss)]:
                print(f'[{epoch:03d}] {loss_name}: {loss_var / len(valdataset):.5f}')
                logger.log({loss_name: loss_var / len(valdataset), 'iter': epoch * len(traindataset), 'epoch': epoch})

            if epoch % config.save_epoch == (config.save_epoch - 1):
                torch.save(model.state_dict(), f'runs/{config.experiment}/model_{epoch}.ckpt')

            for sample_val in valvisdataset:
                sample_val = sample_val.to(device)

                with torch.no_grad():
                    # prediction = model(sample_val.x, sample_val.edge_index, sample_val.num_sub_vertices, sample_val.pool_maps, sample_val.sub_edges)
                    prediction = model(sample_val.x, sample_val.edge_index, sample_val.num_sub_vertices, sample_val.pool_maps, sample_val.pad_sizes, sample_val.sub_edges)
                    prediction = prediction[:sample_val.y.shape[0], :]

                mask = valdataset.mask(sample_val.y).unsqueeze(-1)
                valvisdataset.visualize_graph_with_predictions(sample_val, (sample_val.y * mask).cpu().numpy(), f'runs/{config.experiment}/visualization/epoch_{epoch:05d}', 'gt')
                valvisdataset.visualize_graph_with_predictions(sample_val, (prediction * mask).cpu().numpy(), f'runs/{config.experiment}/visualization/epoch_{epoch:05d}', 'pred')
                valvisdataset.visualize_graph_with_predictions(sample_val, (sample_val.x[:, 3:6] * mask).cpu().numpy(), f'runs/{config.experiment}/visualization/epoch_{epoch:05d}', 'in')

            # set model back to train
            model.train()


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
    GraphNetTrainer(config, logger)


if __name__ == '__main__':
    main()
