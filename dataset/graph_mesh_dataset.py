from pathlib import Path

import hydra
import torch
import os
import random

import trimesh
from torch_geometric.data import Data, Dataset
import numpy as np
from tqdm import tqdm

from util.mesh_proc import mesh_proc
from util.misc import read_list
from util.embedder import get_embedder_nerf


class GraphMeshDataset(Dataset):
    def __init__(self, config, split, use_single_view, transform=None, pre_transform=None):
        splits_file = Path(config.dataset.data_dir) / 'splits' / config.dataset.name / config.dataset.splits_dir / f'{split}.txt'
        self.split_name = f'{config.dataset.splits_dir}_{split}'
        item_list = read_list(splits_file)
        if use_single_view:
            self.items = [(item, 180, 45) for item in item_list]
        else:
            self.items = [(item, x, y) for item in item_list for x in range(225, 60, -45) for y in range(0, 360, 45)]
        # self.items = [x for i, x in enumerate(self.items) if i % config.n_proc == config.proc]
        super().__init__(Path(config.dataset.data_dir, config.dataset.name), transform, pre_transform)

    @staticmethod
    def item_to_name(item):
        return f"{item[0]}_{item[1]:03d}_{item[2]:03d}"

    @property
    def raw_dir(self) -> str:
        return self.root

    @property
    def processed_dir(self) -> str:
        return str(Path(self.root).parent / f'{Path(self.root).name}_{self.split_name}_processed')

    @property
    def raw_file_names(self):
        return [x.name for x in Path(self.raw_dir).iterdir() if x.name in [y[0] for y in self.items]]

    @property
    def processed_file_names(self):
        return [os.path.join(self.processed_dir, self.item_to_name(item)) for item in self.items]

    def download(self):
        pass

    def process(self):
        for x in tqdm(self.items, desc=f'process_{self.split_name}'):
            if not os.path.exists(os.path.join(self.processed_dir, f'{self.item_to_name(x)}.pt')):
                # Read data from `raw_path`.
                data = self.read_mesh_data(Path(self.raw_dir, x[0]) / "model_normalized.obj", Path(self.raw_dir, x[0]) / f"model_normalized_input_{x[1]:03d}_{x[2]:03d}.obj", x[1], x[2])

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, os.path.join(self.processed_dir, f'{self.item_to_name(x)}.pt'))

    def len(self):
        unique_items = list(set([x[0] for x in self.items]))
        return len(unique_items)

    def get(self, idx):
        unique_items = list(set([x[0] for x in self.items]))
        indexed_items = [x for x in self.items if x[0] == unique_items[idx]]
        selected_item = random.choice(indexed_items)
        data = torch.load(os.path.join(self.processed_dir, f'{self.item_to_name(selected_item)}.pt'))
        return data

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

    def visualize_graph_with_predictions(self, item, prediction, output_dir, output_suffix):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        mesh = trimesh.load(Path(self.raw_dir, "_".join(item.name.split('_')[:-2])) / "model_normalized.obj", force='mesh', process=False)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=prediction + 0.5, process=False)
        mesh.export(output_dir / f"{item.name}_{output_suffix}.obj")


@hydra.main(config_path='../config', config_name='graph_nn')
def main(config):
    print(len(GraphMeshDataset(config, 'val', use_single_view=False)))


if __name__ == '__main__':
    main()
