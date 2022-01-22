import json
import shutil
import numpy as np
from pathlib import Path

import torch_scatter
import trimesh
from tqdm import tqdm
import torch
import scipy


processed_shapes_pruned_path = Path('/cluster/gimli/ysiddiqui/CADTextures/Photoshape/shapenet-chairs-manifold-highres-part_processed_color')
mesh_dir = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs")


def add_target_colors():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)

    args = parser.parse_args()
    files = sorted([x for x in processed_shapes_pruned_path.iterdir()])
    files = [x for i, x in enumerate(files) if i % args.num_proc == args.proc]

    for f in tqdm(files):
        pt_arxiv = torch.load(f)
        mesh = trimesh.load(mesh_dir / f.stem / "vcf_model_normalized.obj", process=False)
        vertex_colors = np.array(mesh.visual.vertex_colors, dtype=np.float32)
        face_colors = (vertex_colors[mesh.faces[:, 0], :] + vertex_colors[mesh.faces[:, 1], :] + vertex_colors[mesh.faces[:, 2], :] + vertex_colors[mesh.faces[:, 3], :]) / 4
        pt_arxiv['target_colors'] = torch.from_numpy(face_colors).float() / 255. - 0.5
        torch.save(pt_arxiv, f)


if __name__ == "__main__":
    add_target_colors()
