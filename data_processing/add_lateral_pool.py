import json
import shutil
import numpy as np
from pathlib import Path

import torch_scatter
import trimesh
from tqdm import tqdm
import torch
import scipy


processed_shapes_pruned_path = Path('/cluster/gimli/ysiddiqui/CADTextures/Photoshape/shapenet-chairs-manifold-highres-part_processed')


def add_lateral_pool():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)

    args = parser.parse_args()
    files = sorted([x for x in processed_shapes_pruned_path.iterdir()])
    files = [x for i, x in enumerate(files) if i % args.num_proc == args.proc]

    for f in tqdm(files):
        pt_arxiv = torch.load(f)
        num_sub_vertices = [pt_arxiv['conv_data'][i][0].shape[0] for i in range(1, len(pt_arxiv['conv_data']))]
        pool_maps = pt_arxiv['pool_locations']
        lateral_maps = []
        for level in range(len(num_sub_vertices)):
            ones = torch.ones((pt_arxiv['conv_data'][level][0].shape[0], 1))
            occupancy = torch.zeros((num_sub_vertices[level], ones.shape[1]), dtype=ones.dtype).to(ones.device)
            torch_scatter.scatter_max(ones, pool_maps[level], dim=0, out=occupancy)
            occupied = torch.nonzero(occupancy)[:, 0]
            unoccupied = torch.nonzero(1 - occupancy)[:, 0]
            kdtree = scipy.spatial.cKDTree(pt_arxiv['pos_data'][level + 1][occupied, :].numpy())
            _, indices = kdtree.query(pt_arxiv['pos_data'][level + 1][unoccupied, :].numpy(), k=1)
            selected_occupied = occupied[indices]
            lateral_maps.append(torch.cat([unoccupied.long().unsqueeze(1), selected_occupied.long().unsqueeze(1)], 1))
        pt_arxiv['lateral_maps'] = lateral_maps
        torch.save(pt_arxiv, f)

if __name__ == "__main__":
    add_lateral_pool()
