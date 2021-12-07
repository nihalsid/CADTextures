import json
import trimesh
from pathlib import Path
import numpy as np
import random

from data_processing.process_mesh_for_conv_generalized import calculate_face_hierarchies


def unpool(x, pool_map):
    x_unpooled = x[pool_map, :]
    return x_unpooled


def test_pool_tracing():
    num_samples_per_level = 24
    target_dir = Path("runs/test_pool_trace")
    target_dir.mkdir(exist_ok=True)
    input_dir = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs-manifold/shape02450_rank01_pair100197")
    selections = json.loads((input_dir / "selection.json").read_text())
    faces = sorted(list(selections.keys()), key=lambda x: int(x), reverse=True)
    qmeshes = list(reversed([f"quad_{int(f):05d}_{selections[f]:03d}.obj" for f in faces] + [f'quad_{int(24):05d}_{1:03d}.obj']))
    trace_maps = list(reversed(calculate_face_hierarchies(input_dir)))
    for level in range(len(qmeshes)):
        mesh = trimesh.load(input_dir / qmeshes[level], process=False)
        for idx in range(num_samples_per_level):
            face_colors = np.zeros((mesh.faces.shape[0], 3))
            selected_face = random.randint(0, face_colors.shape[0] - 1)
            face_colors[selected_face, :] = np.array([0, 255, 0])
            (target_dir / f"{mesh.faces.shape[0]:05d}").mkdir(exist_ok=True, parents=True)
            trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_colors=face_colors, process=False).export(target_dir / f"{mesh.faces.shape[0]:05d}" / f"{idx:02d}_{mesh.faces.shape[0]:05d}.obj")
            for trace_level in range(level, len(trace_maps)):
                unpooled_mesh = trimesh.load(input_dir / qmeshes[trace_level + 1], process=False)
                face_colors = unpool(face_colors, trace_maps[trace_level])
                trimesh.Trimesh(vertices=unpooled_mesh.vertices, faces=unpooled_mesh.faces, face_colors=face_colors, process=False).export(target_dir / f"{mesh.faces.shape[0]:05d}" / f"{idx:02d}_{unpooled_mesh.faces.shape[0]:05d}.obj")


if __name__ == '__main__':
    test_pool_tracing()
