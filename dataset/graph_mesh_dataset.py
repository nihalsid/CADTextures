import os
import random
from pathlib import Path

import hydra
import numpy as np
import torch
import trimesh
from torch.utils.data.dataloader import default_collate
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from collections.abc import Mapping, Sequence

from util.mesh_proc import mesh_proc
from util.misc import read_list, DotDict


def to_device(batch, device):
    for k in batch.keys():
        if isinstance(batch[k], torch.Tensor):
            batch[k] = batch[k].to(device)
    for k in batch['graph_data'].keys():
        if isinstance(batch['graph_data'][k], torch.Tensor):
            batch['graph_data'][k] = batch['graph_data'][k].to(device)
        elif isinstance(batch['graph_data'][k], list):
            for m in range(len(batch['graph_data'][k])):
                if isinstance(batch['graph_data'][k][m], torch.Tensor):
                    batch['graph_data'][k][m] = batch['graph_data'][k][m].to(device)
    return batch


class GraphMeshDataset(Dataset):

    def __init__(self, config, split, use_single_view, load_to_memory=False, use_all_views=False):
        splits_file = Path(config.dataset.data_dir) / 'splits' / config.dataset.name / config.dataset.splits_dir / f'{split}.txt'
        self.split_name = f'{config.dataset.splits_dir}_{split}'
        self.use_all_views = use_all_views
        item_list = read_list(splits_file)
        self.plane_dataset = config.dataset.plane
        if not config.dataset.plane:
            if use_single_view:
                self.items = [(item, 180, 45) for item in item_list]
            else:
                self.items = [(item, x, y) for item in item_list for x in range(225, 60, -45) for y in range(0, 360, 45)]
            self.target_name = "model_normalized.obj"
            self.input_name = lambda x, y: f"model_normalized_input_{x:03d}_{y:03d}.obj"
            self.mask = lambda x, bs: torch.ones((x.shape[0],)).float().to(x.device)
            split_name = config.dataset.name.split('/')
            mesh = trimesh.load(Path(config.dataset.data_dir, split_name[0] + '-model', split_name[1]) / "coloredbrodatz_D48_COLORED" / self.target_name, process=False)
            vertex_to_uv = np.array(mesh.visual.uv)
            self.indices_dest_i = []
            self.indices_dest_j = []
            self.indices_src = []
            for v_idx in range(vertex_to_uv.shape[0]):
                j = int(round(vertex_to_uv[v_idx][0] * 127))
                i = 127 - int(round(vertex_to_uv[v_idx][1] * 127))
                self.indices_dest_i.append(i)
                self.indices_dest_j.append(j)
                self.indices_src.append(v_idx)
            self.to_image = self.mesh_to_image
        else:
            if use_single_view:
                self.items = [(item, 0, 0) for item in item_list]
            else:
                self.items = [(item, x, 0) for item in item_list for x in range(12)]
            self.target_name = "surface_texture.obj"
            self.input_name = lambda x, y: f"inv_partial_texture_{x:03d}.obj"
            mesh = trimesh.load(Path(config.dataset.data_dir, config.dataset.name) / "coloredbrodatz_D48_COLORED" / self.target_name, process=False)
            self.target_valid_area = mesh.visual.vertex_colors[:, 3] == 255
            self.sort_indices = np.lexsort((mesh.vertices[:, 0], mesh.vertices[:, 1], mesh.vertices[:, 2]))
            self.mask = lambda x, bs: torch.from_numpy(self.target_valid_area).repeat(bs).float().to(x.device)
            self.to_image = self.plane_to_image
        self.items = [x for i, x in enumerate(self.items) if i % config.n_proc == config.proc]
        super().__init__(Path(config.dataset.data_dir, config.dataset.name))
        self.memory = []
        self.load_to_memory = False
        if load_to_memory:
            for i in tqdm(range(self.__len__()), 'load_to_mem'):
                self.memory.append(self.__getitem__(i))
            self.load_to_memory = True

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
                data = self.read_mesh_data(Path(self.raw_dir, x[0]) / self.target_name, Path(self.raw_dir, x[0]) / self.input_name(x[1], x[2]), x[1], x[2])

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue

                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                torch.save(data, os.path.join(self.processed_dir, f'{self.item_to_name(x)}.pt'))

    def len(self):
        if self.use_all_views:
            if self.plane_dataset:  # just has 12
                return len(self.items)
            else:  # has 32 - don't dump all
                return len(self.items) // 4
        else:
            unique_items = list(set([x[0] for x in self.items]))
            return len(unique_items)

    def get(self, idx):
        if self.load_to_memory:
            return self.memory[idx]
        if self.use_all_views:
            if self.plane_dataset:
                selected_item = self.items[idx]
            else:
                unique_items = sorted(list(set([x[0] for x in self.items])))
                xy = [x[1:] for x in self.items if x[0] == unique_items[0]][::4]
                new_item_list = [(item, x[0], x[1]) for item in unique_items for x in xy]
                selected_item = new_item_list[idx]
        else:
            unique_items = list(set([x[0] for x in self.items]))
            indexed_items = [x for x in self.items if x[0] == unique_items[idx]]
            selected_item = random.choice(indexed_items)
        data = torch.load(os.path.join(self.processed_dir, f'{self.item_to_name(selected_item)}.pt'))
        level_masks = [torch.zeros(shape).long() for shape in [data.x.shape[0]] + data.num_sub_vertices.numpy().tolist()]
        return {
            "name": data.name,
            "x": data.x,
            "y": data.y,
            "graph_data": self.get_item_as_graphdata(data.edge_index, data.sub_edges, data.num_sub_vertices, data.pool_maps, level_masks)
        }

    def read_mesh_data(self, path_all, path_input, x, y):
        nodes, input_colors, valid_colors, target_colors, edges, edge_features, vertex_features, num_sub_vertices, sub_edges, pool_maps = mesh_proc(str(path_all), str(path_input))
        # nodes = -1 + (nodes - nodes.min()) / (nodes.max() - nodes.min()) * 2
        # embedder, embedder_out_dim = get_embedder_nerf(10, input_dims=3, i=0)
        # nodes = embedder(torch.from_numpy(nodes)).numpy()
        if self.plane_dataset:
            x_feat = torch.from_numpy(np.hstack((input_colors, valid_colors.reshape(-1, 1)))).float()
        else:
            x_feat = torch.from_numpy(np.hstack((nodes, input_colors, valid_colors.reshape(-1, 1), vertex_features))).float()
        mesh_data = Data(x=x_feat,
                         y=torch.from_numpy(target_colors).float(),
                         edge_index=torch.from_numpy(np.hstack([edges, edges[[1, 0], :]])).long(),
                         edge_attr=torch.from_numpy(np.vstack([edge_features, edge_features])).float(),
                         num_sub_vertices=torch.from_numpy(np.array(num_sub_vertices)).long(),
                         sub_edges=[torch.from_numpy(x).long() for x in sub_edges],
                         pool_maps=[torch.from_numpy(x).long() for x in pool_maps],
                         name=f"{path_all.parent.name}_{x:03d}_{y:03d}")
        return mesh_data

    def visualize_graph_with_predictions(self, name, prediction, output_dir, output_suffix):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        mesh = trimesh.load(Path(self.raw_dir, "_".join(name.split('_')[:-2])) / self.target_name, force='mesh', process=False)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=prediction + 0.5, process=False)
        mesh.export(output_dir / f"{name}_{output_suffix}.obj")

    def plane_to_image(self, vertex_colors):
        sort_index_i = 0
        image = torch.zeros((3, 128, 128)).float().to(vertex_colors.device)
        for i in range(127, -1, -1):
            for j in range(128):
                image[:, i, j] = vertex_colors[self.sort_indices[sort_index_i], :3]
                sort_index_i += 1
            sort_index_i += 1
        return image

    def mesh_to_image(self, vertex_colors, level_mask):
        batch_size = level_mask.max() + 1
        image = torch.zeros((batch_size, 3, 128, 128), device=vertex_colors.device)
        indices_dest_i = torch.tensor(self.indices_dest_i * batch_size, device=vertex_colors.device).long()
        indices_dest_j = torch.tensor(self.indices_dest_j * batch_size, device=vertex_colors.device).long()
        indices_src = torch.tensor(self.indices_src * batch_size, device=vertex_colors.device).long()
        image[level_mask, :, indices_dest_i, indices_dest_j] = vertex_colors[indices_src + level_mask * len(self.indices_src), :]
        return image

    @staticmethod
    def get_item_as_graphdata(edge_index, sub_edges, num_sub_vertices, pool_maps, level_masks):
        return DotDict({
            'edge_index': edge_index,
            'sub_edges': sub_edges,
            'node_counts': num_sub_vertices,
            'pool_maps': pool_maps,
            'level_masks': level_masks
        })

    @staticmethod
    def batch_mask(t, graph_data, idx, level=0):
        return t[graph_data['level_masks'][level] == idx]


class FaceGraphMeshDataset(torch.utils.data.Dataset):

    def __init__(self, config, split, use_single_view, load_to_memory=False, use_all_views=False):
        self.raw_dir = Path(config.dataset.data_dir, config.dataset.name)
        splits_file = Path(config.dataset.data_dir) / 'splits' / config.dataset.name / config.dataset.splits_dir / f'{split}.txt'
        self.split_name = f'{config.dataset.splits_dir}_{split}'
        self.use_all_views = use_all_views
        item_list = read_list(splits_file)
        self.plane_dataset = config.dataset.plane
        if not config.dataset.plane:
            if use_single_view:
                self.items = [(item, 180, 45) for item in item_list]
            else:
                self.items = [(item, x, y) for item in item_list for x in range(225, 60, -45) for y in range(0, 360, 45)]
            self.target_name = "model_normalized.obj"
            self.input_name = lambda x, y: f"model_normalized_input_{x:03d}_{y:03d}.obj"
            self.mask = lambda x, bs: torch.ones((x.shape[0],)).float().to(x.device)
            split_name = config.dataset.name.split('/')
            mesh = trimesh.load(Path(config.dataset.data_dir, split_name[0] + '-model', split_name[1]) / "coloredbrodatz_D48_COLORED" / self.target_name, process=False)
            vertex_to_uv = np.array(mesh.visual.uv)
            faces_to_vertices = np.array(mesh.faces)
            a = vertex_to_uv[faces_to_vertices[:, 0], :]
            b = vertex_to_uv[faces_to_vertices[:, 1], :]
            c = vertex_to_uv[faces_to_vertices[:, 2], :]
            d = vertex_to_uv[faces_to_vertices[:, 3], :]
            self.faces_to_uv = (a + b + c + d) / 4
            self.indices_dest_i = []
            self.indices_dest_j = []
            self.indices_src = []
            for v_idx in range(self.faces_to_uv.shape[0]):
                j = int(round(self.faces_to_uv[v_idx][0] * 127))
                i = 127 - int(round(self.faces_to_uv[v_idx][1] * 127))
                self.indices_dest_i.append(i)
                self.indices_dest_j.append(j)
                self.indices_src.append(v_idx)
        else:
            if use_single_view:
                self.items = [(item, 0, 0) for item in item_list]
            else:
                self.items = [(item, x, 0) for item in item_list for x in range(12)]
            self.target_name = "surface_texture.obj"
            self.input_name = lambda x, y: f"inv_partial_texture_{x:03d}.obj"
            mesh = trimesh.load(Path(config.dataset.data_dir, config.dataset.name) / "coloredbrodatz_D48_COLORED" / self.target_name, process=False)
            self.target_valid_area = mesh.visual.face_colors[:, 3] == 255
            mesh_triangle_center = mesh.triangles.mean(axis=1)
            sort_indices = np.lexsort((mesh_triangle_center[:, 0], mesh_triangle_center[:, 1], mesh_triangle_center[:, 2]))
            self.mask = lambda x, bs: torch.from_numpy(self.target_valid_area).repeat(bs).float().to(x.device)
            sort_index_i = 0
            self.indices_dest_i = []
            self.indices_dest_j = []
            self.indices_src = []
            for i in range(127, -1, -1):
                for j in range(128):
                    self.indices_dest_i.append(i)
                    self.indices_dest_j.append(j)
                    self.indices_src.append(sort_indices[sort_index_i])
                    sort_index_i += 1
        self.items = [x for i, x in enumerate(self.items) if i % config.n_proc == config.proc]
        self.memory = []
        self.load_to_memory = False
        if load_to_memory:
            for i in tqdm(range(self.__len__()), 'load_to_mem'):
                self.memory.append(self.__getitem__(i))
            self.load_to_memory = True

    @staticmethod
    def item_to_name(item):
        return f"{item[0]}_{item[1]:03d}_{item[2]:03d}"

    @property
    def processed_dir(self) -> str:
        return str(Path(self.raw_dir).parent / f'{Path(self.raw_dir).name}_FC_processed')

    def __len__(self):
        if self.use_all_views:
            if self.plane_dataset:  # just has 12
                return len(self.items)
            else:  # has 32 - don't dump all
                return len(self.items) // 4
        else:
            unique_items = list(set([x[0] for x in self.items]))
            return len(unique_items)

    def __getitem__(self, idx):
        if self.load_to_memory:
            return self.memory[idx]
        if self.use_all_views:
            if self.plane_dataset:
                selected_item = self.items[idx]
            else:
                unique_items = sorted(list(set([x[0] for x in self.items])))
                xy = [x[1:] for x in self.items if x[0] == unique_items[0]][::4]
                new_item_list = [(item, x[0], x[1]) for item in unique_items for x in xy]
                selected_item = new_item_list[idx]
        else:
            unique_items = list(set([x[0] for x in self.items]))
            indexed_items = [x for x in self.items if x[0] == unique_items[idx]]
            selected_item = random.choice(indexed_items)
        pt_arxiv = torch.load(os.path.join(self.processed_dir, f'{self.item_to_name(selected_item)}.pt'))
        if self.plane_dataset:
            x_feat = torch.cat((pt_arxiv['input_positions'], pt_arxiv['input_colors'], pt_arxiv['valid_input_colors'].reshape(-1, 1)), 1).float()
        else:
            x_feat = torch.cat((pt_arxiv['input_positions'], pt_arxiv['input_colors'], pt_arxiv['valid_input_colors'].reshape(-1, 1), pt_arxiv['input_normals']), 1).float()

        edge_index = pt_arxiv['conv_data'][0][0].long()
        num_sub_vertices = [pt_arxiv['conv_data'][i][0].shape[0] for i in range(1, len(pt_arxiv['conv_data']))]
        pad_sizes = [pt_arxiv['conv_data'][i][2].shape[0] for i in range(len(pt_arxiv['conv_data']))]
        sub_edges = [pt_arxiv['conv_data'][i][0].long() for i in range(1, len(pt_arxiv['conv_data']))]
        pool_maps = pt_arxiv['pool_locations']
        is_pad = [pt_arxiv['conv_data'][i][4].bool() for i in range(len(pt_arxiv['conv_data']))]
        level_masks = [torch.zeros(pt_arxiv['conv_data'][i][0].shape[0]).long() for i in range(len(pt_arxiv['conv_data']))]
        return {
            "name": f"{self.item_to_name(selected_item)}",
            "x": x_feat,
            "y": pt_arxiv['target_colors'].float(),
            "graph_data": self.get_item_as_graphdata(edge_index, sub_edges, pad_sizes, num_sub_vertices, pool_maps, is_pad, level_masks)
        }

    @staticmethod
    def get_item_as_graphdata(edge_index, sub_edges, pad_sizes, num_sub_vertices, pool_maps, is_pad, level_masks):
        return DotDict({
            'face_neighborhood': edge_index,
            'sub_neighborhoods': sub_edges,
            'pads': pad_sizes,
            'node_counts': num_sub_vertices,
            'pool_maps': pool_maps,
            'is_pad': is_pad,
            'level_masks': level_masks
        })

    def visualize_graph_with_predictions(self, name, prediction, output_dir, output_suffix):
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        mesh = trimesh.load(Path(self.raw_dir, "_".join(name.split('_')[:-2])) / self.target_name, force='mesh', process=False)
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_colors=prediction + 0.5, process=False)
        mesh.export(output_dir / f"{name}_{output_suffix}.obj")

    def to_image(self, face_colors, level_mask):
        batch_size = level_mask.max() + 1
        image = torch.zeros((batch_size, 3, 128, 128), device=face_colors.device)
        indices_dest_i = torch.tensor(self.indices_dest_i * batch_size, device=face_colors.device).long()
        indices_dest_j = torch.tensor(self.indices_dest_j * batch_size, device=face_colors.device).long()
        indices_src = torch.tensor(self.indices_src * batch_size, device=face_colors.device).long()
        image[level_mask, :, indices_dest_i, indices_dest_j] = face_colors[indices_src + level_mask * len(self.indices_src), :]
        return image

    @staticmethod
    def batch_mask(t, graph_data, idx, level=0):
        return t[graph_data['level_masks'][level] == idx]


class GraphDataLoader(torch.utils.data.DataLoader):

    def __init__(
            self,
            dataset: Dataset,
            batch_size: int = 1,
            shuffle: bool = False,
            follow_batch=[],
            exclude_keys=[],
            **kwargs,
    ):
        if "collate_fn" in kwargs:
            del kwargs["collate_fn"]

        # Save for PyTorch Lightning...
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

        super().__init__(dataset, batch_size, shuffle, collate_fn=Collater(follow_batch, exclude_keys), **kwargs)


class Collater(object):

    def __init__(self, follow_batch, exclude_keys):
        self.follow_batch = follow_batch
        self.exclude_keys = exclude_keys

    @staticmethod
    def cat_collate(batch, dim=0):
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            out = None
            if torch.utils.data.get_worker_info() is not None:
                # If we're in a background process, concatenate directly into a
                # shared memory tensor to avoid an extra copy
                numel = sum([x.numel() for x in batch])
                storage = elem.storage()._new_shared(numel)
                out = elem.new(storage)
            return torch.cat(batch, dim, out=out)
        raise NotImplementedError

    def collate(self, batch):
        elem = batch[0]
        if isinstance(elem, torch.Tensor):
            return default_collate(batch)
        elif isinstance(elem, float):
            return torch.tensor(batch, dtype=torch.float)
        elif isinstance(elem, int):
            return torch.tensor(batch)
        elif isinstance(elem, str):
            return batch
        elif isinstance(elem, DotDict):
            if 'face_neighborhood' in elem:  # face conv data
                face_neighborhood, sub_neighborhoods, pads, node_counts, pool_maps, is_pad, level_masks = [], [], [], [], [], [], []
                pad_sum = 0

                for b_i in range(len(batch)):
                    face_neighborhood.append(batch[b_i].face_neighborhood + pad_sum)
                    pad_sum += batch[b_i].is_pad[0].shape[0]

                for sub_i in range(len(elem.pads)):
                    is_pad_n = []
                    pad_n = 0
                    for b_i in range(len(batch)):
                        is_pad_n.append(batch[b_i].is_pad[sub_i])
                        batch[b_i].level_masks[sub_i][:] = b_i
                        pad_n += batch[b_i].pads[sub_i]
                    is_pad.append(self.cat_collate(is_pad_n))
                    level_masks.append(self.cat_collate([batch[b_i].level_masks[sub_i] for b_i in range(len(batch))]))
                    pads.append(pad_n)

                for sub_i in range(len(elem.sub_neighborhoods)):
                    sub_n = []
                    pool_n = []
                    node_count_n = 0
                    pad_sum = 0
                    for b_i in range(len(batch)):
                        sub_n.append(batch[b_i].sub_neighborhoods[sub_i] + pad_sum)
                        pool_n.append(batch[b_i].pool_maps[sub_i] + node_count_n)
                        pad_sum += batch[b_i].is_pad[sub_i + 1].shape[0]
                        node_count_n += batch[b_i].node_counts[sub_i]
                    sub_neighborhoods.append(self.cat_collate(sub_n))
                    node_counts.append(node_count_n)
                    pool_maps.append(self.cat_collate(pool_n))

                batch_dot_dict = {
                    'face_neighborhood': self.cat_collate(face_neighborhood),
                    'sub_neighborhoods': sub_neighborhoods,
                    'pads': pads,
                    'node_counts': node_counts,
                    'pool_maps': pool_maps,
                    'is_pad': is_pad,
                    'level_masks': level_masks
                }

                return batch_dot_dict
            else:  # graph conv data
                edge_index, sub_edges, node_counts, pool_maps, level_masks = [], [], [], [], []
                pad_sum = 0
                for b_i in range(len(batch)):
                    edge_index.append(batch[b_i].edge_index + pad_sum)
                    pad_sum += batch[b_i].pool_maps[0].shape[0]

                for sub_i in range(len(elem.sub_edges) + 1):
                    for b_i in range(len(batch)):
                        batch[b_i].level_masks[sub_i][:] = b_i
                    level_masks.append(self.cat_collate([batch[b_i].level_masks[sub_i] for b_i in range(len(batch))]))

                for sub_i in range(len(elem.sub_edges)):
                    sub_n = []
                    pool_n = []
                    node_count_n = 0
                    for b_i in range(len(batch)):
                        sub_n.append(batch[b_i].sub_edges[sub_i] + node_count_n)
                        pool_n.append(batch[b_i].pool_maps[sub_i] + node_count_n)
                        node_count_n += batch[b_i].node_counts[sub_i]
                    sub_edges.append(self.cat_collate(sub_n, dim=1))
                    node_counts.append(node_count_n)
                    pool_maps.append(self.cat_collate(pool_n))

                batch_dot_dict = {
                    'edge_index': self.cat_collate(edge_index, dim=1),
                    'sub_edges': sub_edges,
                    'node_counts': node_counts,
                    'pool_maps': pool_maps,
                    'level_masks': level_masks
                }

                return batch_dot_dict
        elif isinstance(elem, Mapping):
            retdict = {}
            for key in elem:
                if key in ['x', 'y']:
                    retdict[key] = self.cat_collate([d[key] for d in batch])
                else:
                    retdict[key] = self.collate([d[key] for d in batch])
            return retdict
        elif isinstance(elem, tuple) and hasattr(elem, '_fields'):
            return type(elem)(*(self.collate(s) for s in zip(*batch)))
        elif isinstance(elem, Sequence) and not isinstance(elem, str):
            return [self.collate(s) for s in zip(*batch)]
        raise TypeError('DataLoader found invalid type: {}'.format(type(elem)))

    def __call__(self, batch):
        return self.collate(batch)


@hydra.main(config_path='../config', config_name='graph_nn')
def main(config):
    print(len(GraphMeshDataset(config, 'val', use_single_view=False)))


if __name__ == '__main__':
    main()
