import hydra
import torch
import trimesh

from dataset.graph_mesh_dataset import FaceGraphMeshDataset, GraphDataLoader
import tqdm
from PIL import Image
import numpy as np

from model.graphnet import pool, unpool, SpatialAttentionConv, BigFaceEncoderDecoder, WrappedLinear


@hydra.main(config_path='../config', config_name='graph_nn_test')
def test_loader(config, test_models=True, test_reconstruction=False, test_decimation=False, visualize_images=False):
    batch_size = 4
    conv = SpatialAttentionConv(in_channels=10, out_channels=3)
    model = BigFaceEncoderDecoder(10, 3, 128, SpatialAttentionConv, num_pools=config.dataset.num_pools, input_transform=WrappedLinear)
    dataset = FaceGraphMeshDataset(config, 'val_vis', use_single_view=True, load_to_memory=False)
    dataloader = GraphDataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mesh_decimations = ["model_normalized"] + [f"decimate_{i}" for i in range(1, config.dataset.num_pools + 1)]
    meshes = [trimesh.load(f"data/{config.dataset.name}/coloredbrodatz_D48_COLORED/{x}.obj", process=False) for x in mesh_decimations]
    for idx, batch in enumerate(dataloader):
        # Test data member sizes
        print(f"batch_idx {idx:02d}: ")
        print(f'\tx: {batch["x"].shape}\n\ty: {batch["y"].shape}')
        for key in batch['graph_data']:
            print(f'\t{key}: {len(batch["graph_data"][key])}')
        # Test conv & model
        if test_models:
            # print(conv(batch['x'], batch['graph_data']['face_neighborhood'], batch['graph_data']['is_pad'][0], batch['graph_data']['pads'][0]).shape)
            print(model(batch['x'], batch['graph_data']).shape)
        # Test pool(3) + unpool(3)
        if test_reconstruction:
            y = pool(batch["y"] + 0.5, batch["graph_data"]["node_counts"][0], batch["graph_data"]["pool_maps"][0], pool_op='mean')
            y = pool(y, batch["graph_data"]["node_counts"][1], batch["graph_data"]["pool_maps"][1], pool_op='mean')
            y = pool(y, batch["graph_data"]["node_counts"][2], batch["graph_data"]["pool_maps"][2], pool_op='mean')
            y = unpool(y, batch["graph_data"]["pool_maps"][2])
            y = unpool(y, batch["graph_data"]["pool_maps"][1])
            y = unpool(y, batch["graph_data"]["pool_maps"][0])
            for bi in range(batch_size):
                batch_yi_masked = dataset.batch_mask(y, batch['graph_data'], bi, level=0)
                trimesh.Trimesh(vertices=meshes[0].vertices, faces=meshes[0].faces, face_colors=batch_yi_masked.numpy(), process=False).export(f'reconstruct_{bi}.obj')
        # Test decimation pooling and level masking
        if test_decimation:
            face_colors_0 = batch["y"].numpy() + 0.5
            for mi in range(len(meshes) - 1):
                face_colors_1 = pool(torch.from_numpy(face_colors_0).float(), batch["graph_data"]["node_counts"][mi], batch["graph_data"]["pool_maps"][mi], pool_op='mean').numpy()
                for bi in range(batch_size):
                    batch_yi_masked = dataset.batch_mask(face_colors_1, batch['graph_data'], bi, level=mi + 1)
                    trimesh.Trimesh(vertices=meshes[mi + 1].vertices, faces=meshes[mi + 1].faces, face_colors=batch_yi_masked, process=False).export(f'decimation_{bi}_{mi + 1}.obj')
                face_colors_0 = face_colors_1.copy()
        # Test x, y visualizations
        if visualize_images:
            for i in range(batch_size):
                batch_yi_masked = dataset.batch_mask(batch['y'], batch['graph_data'], i)
                batch_xi_masked = dataset.batch_mask(batch['x'], batch['graph_data'], i)
                mask = dataset.mask(batch_yi_masked)
                target_as_image = dataset.to_image((batch_yi_masked * mask.unsqueeze(-1))).unsqueeze(0)
                input_as_image = dataset.to_image((batch_xi_masked[:, 3:6] * mask.unsqueeze(-1))).unsqueeze(0)
                Image.fromarray(((input_as_image.squeeze(0).permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)).save(f'input_{i}.png')
                Image.fromarray(((target_as_image.squeeze(0).permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)).save(f'target_{i}.png')


@hydra.main(config_path='../config', config_name='graph_nn_test')
def test_memory(config):
    dataset = FaceGraphMeshDataset(config, 'val_vis', use_single_view=True, load_to_memory=False)
    dataloader = GraphDataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    for _ in tqdm.tqdm(range(1000)):
        for data in dataloader:
            pass


if __name__ == "__main__":
    test_loader()
