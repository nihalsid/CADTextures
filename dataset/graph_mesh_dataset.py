from pathlib import Path

import hydra
import torch
import os

import trimesh
from torch_geometric.data import Data, InMemoryDataset
import numpy as np
from tqdm import tqdm

from util.mesh_proc import mesh_proc
from util.misc import read_list
from util.embedder import get_embedder_nerf


class GraphMeshDataset(InMemoryDataset):

    def __init__(self, config, split, use_single_view=False, transform=None, pre_transform=None):
        splits_file = Path(config.dataset.data_dir) / 'splits' / config.dataset.name / config.dataset.splits_dir / f'{split}.txt'
        self.split_name = f'{config.dataset.splits_dir}_{split}'
        item_list = read_list(splits_file)
        if use_single_view:
            self.items = [(item, 180, 45) for item in item_list]
        else:
            self.items = [(item, x, y) for item in item_list for x in range(225, 60, -45) for y in range(0, 360, 45)]
        super().__init__(Path(config.dataset.data_dir, config.dataset.name), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @staticmethod
    def read_mesh_data(path_all, path_input, x, y):
        nodes, input_colors, valid_colors, target_colors, edges, edge_features, vertex_features = mesh_proc(str(path_all), str(path_input))
        nodes = -1 + (nodes - nodes.min()) / (nodes.max() - nodes.min()) * 2
        embedder, embedder_out_dim = get_embedder_nerf(10, input_dims=3, i=0)
        nodes = embedder(torch.from_numpy(nodes)).numpy()
        mesh_data = Data(x=torch.from_numpy(np.hstack((nodes, input_colors, valid_colors.reshape(-1, 1), vertex_features))).float(),
        # mesh_data = Data(x=torch.from_numpy(nodes).float(),
        # mesh_data = Data(x=torch.from_numpy(np.hstack((nodes, input_colors, valid_colors.reshape(-1, 1)))).float(),
                         y=torch.from_numpy(target_colors).float(),
                         edge_index=torch.from_numpy(np.hstack([edges, edges[[1, 0], :]])).long(),
                         edge_attr=torch.from_numpy(np.vstack([edge_features, edge_features])).float(),
                         name=f"{path_all.parent.name}_{x:03d}_{y:03d}")
        return mesh_data

    @property
    def raw_dir(self) -> str:
        return os.path.join(self.root)

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root).parent / f'{Path(self.root).name}_{self.split_name}_processed')

    @property
    def processed_file_names(self):
        return ['processed.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        for x in tqdm(self.items, desc=f'process_{self.split_name}'):
            data_list.append(self.read_mesh_data(Path(self.raw_dir, x[0]) / "model_normalized.obj", Path(self.raw_dir, x[0]) / f"model_normalized_input_{x[1]:03d}_{x[2]:03d}.obj", x[1], x[2]))

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

    @property
    def raw_file_names(self):
        return [x.name for x in Path(self.raw_dir).iterdir() if x.name in self.items]

    def visualize_graph_with_predictions(self, item, prediction, output_dir, output_suffix):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        mesh = trimesh.load(Path(self.raw_dir, "_".join(item.name.split('_')[:-2])) / "model_normalized.obj", force='mesh', process=False)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=prediction + 0.5, process=False)
        mesh.export(output_dir / f"{item.name}_{output_suffix}.obj")


@hydra.main(config_path='../config', config_name='graph_nn')
def main(config):
    print(len(GraphMeshDataset(config, 'train', use_single_view=True)))


if __name__ == '__main__':
    main()
