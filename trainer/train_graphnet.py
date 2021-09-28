import random
from pathlib import Path
from random import randint

import hydra
import wandb
from pytorch_lightning import seed_everything
from pytorch_lightning.loggers import WandbLogger
from tqdm import tqdm

from dataset.graph_mesh_dataset import GraphMeshDataset
import torch
from model.graphnet import GATNet, GraphSAGENet, GCNNet, GraphUNet, GraphSAGEEncoderDecoder
from util.misc import print_model_parameter_count
from util.regression_loss import RegressionLossHelper


def GATNetTrainer(config, logger):

    logger.log_hyperparams(config)

    # declare device
    device = torch.device('cuda:0')

    # instantiate model
    # model = GATNet(63 + 3 + 1 + 6, 3, 256, 0)
    # model = GraphSAGENet(3 + 3 + 1 + 6, 3, 256, 0)
    model = GraphSAGEEncoderDecoder(3 + 3 + 1 + 6, 3, 64)
    # model = GCNNet(63 + 3 + 1 + 6, 3, 256, 0)
    # wandb.watch(model, log='all')

    print_model_parameter_count(model)

    # create dataloaders
    trainset = GraphMeshDataset(config, 'train', use_single_view=config.dataset.single_view, load_to_memory=config.dataset.memory)

    valset = GraphMeshDataset(config, 'val', use_single_view=config.dataset.single_view)

    valvisset = GraphMeshDataset(config, 'val_vis', use_single_view=True)

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

    # declare optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # set model to train, important if your network has e.g. dropout or batchnorm layers
    model.train()

    # keep track of running average of train loss for printing
    train_loss_running = 0.

    for epoch in range(config.max_epoch):
        logger.log_metrics({'epoch': epoch}, epoch * len(traindataset))

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
            prediction = model(sample.x, sample.edge_index, sample.num_sub_vertices, sample.pool_maps, sample.sub_edges)

            # compute loss
            loss_total = loss_criterion.calculate_loss(prediction[:sample.y.shape[0], :], sample.y).mean(dim=1).mean()

            # compute gradients on loss_total
            loss_total.backward()

            # update network params
            optimizer.step()

            # loss logging
            train_loss_running += loss_total.item()
            iteration = epoch * len(traindataset) + iter_idx

            if iteration % config.print_interval == (config.print_interval - 1):
                last_train_loss = train_loss_running / config.print_interval
                logger.log_metrics({'loss_train': last_train_loss}, iteration)
                train_loss_running = 0.
                pbar.set_postfix({'loss': f'{last_train_loss:.4f}'})

        # validation evaluation and logging
        if epoch % config.val_check_interval == (config.val_check_interval - 1):

            # set model to eval, important if your network has e.g. dropout or batchnorm layers
            model.eval()

            loss_total_val = 0

            # forward pass and evaluation for entire validation set
            for sample_val in valdataset:

                sample_val = sample_val.to(device)

                with torch.no_grad():
                    prediction = model(sample_val.x, sample_val.edge_index, sample_val.num_sub_vertices, sample_val.pool_maps, sample_val.sub_edges)
                    prediction = prediction[:sample_val.y.shape[0], :]

                loss_total_val += l1_criterion.calculate_loss(prediction, sample_val.y).mean().item()

            print(f'[{epoch:03d}] val_loss: {loss_total_val / len(valdataset):.3f}')
            logger.log_metrics({'loss_val': loss_total_val / len(valdataset)}, epoch * len(traindataset))

            if epoch % config.save_epoch == (config.save_epoch - 1):
                torch.save(model.state_dict(), f'runs/{config.experiment}/model_{epoch}.ckpt')

            for sample_val in valvisdataset:
                sample_val = sample_val.to(device)

                with torch.no_grad():
                    prediction = model(sample_val.x, sample_val.edge_index, sample_val.num_sub_vertices, sample_val.pool_maps, sample_val.sub_edges)
                    prediction = prediction[:sample_val.y.shape[0], :]

                valvisdataset.visualize_graph_with_predictions(sample_val, sample_val.y.cpu().numpy(), f'runs/{config.experiment}/visualization/epoch_{epoch:05d}', 'gt')
                valvisdataset.visualize_graph_with_predictions(sample_val, prediction.cpu().numpy(), f'runs/{config.experiment}/visualization/epoch_{epoch:05d}', 'pred')
                valvisdataset.visualize_graph_with_predictions(sample_val, sample_val.x[:, -10:-7].cpu().numpy(), f'runs/{config.experiment}/visualization/epoch_{epoch:05d}', 'in')

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
    logger = WandbLogger(project=f'GATNet{config.suffix}[{ds_name}]', name=config.experiment, id=config.experiment)
    seed_everything(config.seed)
    GATNetTrainer(config, logger)


if __name__ == '__main__':
    main()
