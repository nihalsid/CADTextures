import torch
from model.graphnet import pool, unpool
import trimesh


def test_decimation():
    original = trimesh.load("data/SingleShape/CubeTexturePlane/coloredbrodatz_D48_COLORED/surface_texture.obj", force='mesh', process=False)
    decimate_2 = trimesh.load("data/SingleShape/CubeTexturePlane/coloredbrodatz_D48_COLORED/decimate_2.obj", force='mesh', process=False)
    data = torch.load("data/SingleShape/CubeTexturePlane_overfit_004_val_vis_processed/coloredbrodatz_D48_COLORED_000_000.pt")

    x = pool(data.y, data.num_sub_vertices[0], data.pool_maps[0], pool_op='max')
    x = pool(x, data.num_sub_vertices[1], data.pool_maps[1], pool_op='max')
    trimesh.Trimesh(vertices=decimate_2.vertices, vertex_colors=x[:, :].numpy() + 0.5, faces=decimate_2.faces, process=False).export('decimate_2.obj')
    x = unpool(x, data.pool_maps[1])
    x = unpool(x, data.pool_maps[0])
    trimesh.Trimesh(vertices=original.vertices, vertex_colors=x[:, :].numpy() + 0.5, faces=original.faces, process=False).export('original.obj')
    trimesh.Trimesh(vertices=original.vertices, vertex_colors=data.y.numpy() + 0.5, faces=original.faces, process=False).export('originaly.obj')

    x = pool(torch.ones((data.y.shape[0], 1)), data.num_sub_vertices[0], data.pool_maps[0], pool_op='max')
    x = pool(x, data.num_sub_vertices[1], data.pool_maps[1], pool_op='max')
    trimesh.Trimesh(vertices=decimate_2.vertices, vertex_colors=torch.cat((x, x, x), dim=1).numpy(), faces=decimate_2.faces, process=False).export('decimate_2_mask.obj')


if __name__ == '__main__':
    test_decimation()
