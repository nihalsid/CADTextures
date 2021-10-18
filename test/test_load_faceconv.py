import torch
import trimesh
import numpy as np

from model.graphnet import pool


def test_load_colors():
    pt_arxiv = torch.load("data/SingleShape/CubeTexturePlane_FC_proccesed/coloredbrodatz_D48_COLORED_000_000.pt")
    mesh = trimesh.load("data/SingleShape/CubeTexturePlane/coloredbrodatz_D48_COLORED/surface_texture.obj", process=False)
    trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, face_colors=np.repeat(pt_arxiv['valid_target_colors'].reshape((-1, 1)), 3, 1)).export('test.obj')


def test_decimation_pooling():
    pt_arxiv = torch.load("data/SingleShape/CubeTexturePlane_FC_proccesed/coloredbrodatz_D48_COLORED_000_000.pt")
    mesh_decimations = ["surface_texture"] + [f"decimate_{i}" for i in range(1, 6)]
    meshes = [trimesh.load(f"data/SingleShape/CubeTexturePlane/coloredbrodatz_D48_COLORED/{x}.obj", process=False) for x in mesh_decimations]
    face_colors_0 = pt_arxiv['target_colors'].numpy() + 0.5
    for i in range(len(meshes) - 1):
        face_colors_1 = pool(torch.from_numpy(face_colors_0).float(), meshes[i + 1].faces.shape[0], pt_arxiv['pool_locations'][i], pool_op='mean').numpy()
        trimesh.Trimesh(vertices=meshes[i + 1].vertices, faces=meshes[i + 1].faces, face_colors=face_colors_1).export(f'decimation_{i + 1}.obj')
        face_colors_0 = face_colors_1.copy()


if __name__ == '__main__':
    test_decimation_pooling()
