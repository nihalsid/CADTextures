import torch
import trimesh

from model.graphnet import pool, unpool, Blur


def test_blur_pooling():
    pt_arxiv = torch.load("data/SingleShape/CubeTexturePlaneQuad_FC_processed/coloredbrodatz_D48_COLORED_000_000.pt")
    mesh_decimations = ["surface_texture"] + [f"decimate_{i}" for i in range(1, 6)]
    meshes = [trimesh.load(f"data/SingleShape/CubeTexturePlaneQuad/coloredbrodatz_D48_COLORED/{x}.obj", process=False) for x in mesh_decimations]
    blur = Blur(3)

    y = pool(pt_arxiv['target_colors'] + 0.5, pt_arxiv['conv_data'][1][0].shape[0], pt_arxiv['pool_locations'][0], pool_op='mean')
    y = pool(y, pt_arxiv['conv_data'][2][0].shape[0], pt_arxiv['pool_locations'][1], pool_op='mean')
    y = pool(y, pt_arxiv['conv_data'][3][0].shape[0], pt_arxiv['pool_locations'][2], pool_op='mean')
    y = unpool(y, pt_arxiv['pool_locations'][2])
    y = blur(y, pt_arxiv['conv_data'][2][0].long(), pt_arxiv['conv_data'][2][4].bool(), pt_arxiv['conv_data'][2][2].shape[0])
    y = unpool(y, pt_arxiv['pool_locations'][1])
    y = blur(y, pt_arxiv['conv_data'][1][0].long(), pt_arxiv['conv_data'][1][4].bool(), pt_arxiv['conv_data'][1][2].shape[0])
    y = unpool(y, pt_arxiv['pool_locations'][0])
    y = blur(y, pt_arxiv['conv_data'][0][0].long(), pt_arxiv['conv_data'][0][4].bool(), pt_arxiv['conv_data'][0][2].shape[0])

    trimesh.Trimesh(vertices=meshes[0].vertices, faces=meshes[0].faces, face_colors=y.numpy(), process=False).export(f'reconstruct.obj')


if __name__ == '__main__':
    test_blur_pooling()
