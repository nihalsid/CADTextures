from datetime import datetime
from pathlib import Path

import imageio
import pyrender
import numpy as np
import trimesh
from pyrender import RenderFlags
import logging
import json
import random
import torch
from tqdm import tqdm

# setup logging

from util.camera import spherical_coord_to_cam

formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger('subdivide')
logger.setLevel(logging.DEBUG)
fh = logging.FileHandler(f'/tmp/subdivide_{datetime.now().strftime("%d%m%H%M%S")}.log')
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(fh)
logger.addHandler(ch)


def meta_to_pair(c):
    return f'shape{c["shape_id"]:05d}_rank{(c["rank"] - 1):02d}_pair{c["id"]}'


def load_pair_meta_views(image_path, pairmeta_path):
    dataset_images = [x.stem for x in image_path.iterdir()]
    loaded_json = json.loads(Path(pairmeta_path).read_text())
    ret_dict = {}
    for k in loaded_json.keys():
        if meta_to_pair(loaded_json[k]) in dataset_images:
            ret_dict[meta_to_pair(loaded_json[k])] = loaded_json[k]
    return ret_dict


def render_with_photoshape_view(mesh, view):
    width = height = 256
    spherical_camera = spherical_coord_to_cam(view['fov'], view['azimuth'], view['elevation'])
    camera_pose = np.linalg.inv(spherical_camera.view_mat())
    r = pyrender.OffscreenRenderer(width, height)
    camera = pyrender.PerspectiveCamera(yfov=np.pi * view['fov'] / 180, aspectRatio=1.0, znear=0.001)
    camera_intrinsics = np.eye(4, dtype=np.float32)
    camera_intrinsics[0, 0] = camera_intrinsics[1, 1] = height / (2 * np.tan(camera.yfov / 2.0))
    camera_intrinsics[0, 2] = camera_intrinsics[1, 2] = camera_intrinsics[2, 2] = -height / 2
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.5, 0.5, 0.5])
    geo = pyrender.Mesh.from_trimesh(mesh)
    scene.add(geo)
    scene.add(camera, pose=camera_pose)
    color_flat, depth = r.render(scene, flags=RenderFlags.FLAT | RenderFlags.SKIP_CULL_FACES | RenderFlags.RGBA)
    return color_flat, depth.astype(np.float32), camera_intrinsics[:3, :].astype(np.float32), spherical_camera.view_mat()[:3, :].astype(np.float32)


def render_with_photoshape_views(proc, num_proc):
    pairmeta_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/metadata/pairs.json")
    image_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/exemplars")
    mesh_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/shapenet-chairs-manifold-highres")
    output_path = Path(f"/cluster/gimli/ysiddiqui/CADTextures/Photoshape/tex_field")
    output_path.mkdir(exist_ok=True)
    views = load_pair_meta_views(image_path, pairmeta_path)
    view_keys = sorted(list(views.keys()))
    mesh_paths = [x for i, x in enumerate(mesh_path.iterdir()) if i % num_proc == proc]

    for mp in tqdm(mesh_paths):
        (output_path / mp.name / "depth").mkdir(exist_ok=True, parents=True)
        depths, cameras = [], {}
        mesh = trimesh.load(mp / "model_normalized.obj", process=True)
        sample_points, face_indices = trimesh.sample.sample_surface(mesh, count=100000)
        np.savez_compressed(str(output_path / mp.name / "pointcloud.npz"),
                            points=sample_points,
                            normals=np.array(mesh.face_normals[face_indices, :]),
                            loc=np.array([0.0, 0.0, 0.0]), scale=np.array([1.0]))
        for ctr in range(28):
            image, depth, camera_k, camera_w = render_with_photoshape_view(mesh, views[random.choice(view_keys)])
            cameras[f"camera_mat_{ctr}"] = camera_k
            cameras[f"world_mat_{ctr}"] = camera_w
            np.save(str(output_path / mp.name / "depth" / f"{ctr:03d}.npy"), depth)
        np.savez_compressed(str(output_path / mp.name / "depth" / "cameras.npz"), **cameras)


def visualize_depth(path):
    camera = np.load(str(Path(path) / "depth" / "cameras.npz"))
    loc3ds = []
    for p in (Path(path) / "depth").iterdir():
        if p.name.endswith('.exr'):
            depth = imageio.imread(p)
            camera_k = camera[f'camera_mat_{int(p.stem)}']
            camera_w = camera[f'world_mat_{int(p.stem)}']
            loc3d, mask = depth_map_to_3d(torch.from_numpy(depth).unsqueeze(0).unsqueeze(0), torch.from_numpy(camera_k).unsqueeze(0), torch.from_numpy(camera_w).unsqueeze(0))
            loc3d = loc3d.permute((0, 2, 3, 1)).reshape((-1, 3))
            masked_loc3d = loc3d[mask.permute((0, 2, 3, 1)).flatten() == 1,:]
            loc3ds.append(masked_loc3d)
    (Path(path) / "loc3d.obj").write_text("\n".join([f"v {ll[0]} {ll[1]} {ll[2]}" for ll in torch.cat(loc3ds, dim=0).numpy()]))


def depth_map_to_3d(depth, cam_K, cam_W):
    """Derive 3D locations of each pixel of a depth map.

    Args:
        depth (torch.FloatTensor): tensor of size B x 1 x N x M
            with depth at every pixel
        cam_K (torch.FloatTensor): tensor of size B x 3 x 4 representing
            camera matrices
        cam_W (torch.FloatTensor): tensor of size B x 3 x 4 representing
            world matrices
    Returns:
        loc3d (torch.FloatTensor): tensor of size B x 3 x N x M
            representing color at given 3d locations
        mask (torch.FloatTensor):  tensor of size B x 1 x N x M with
            a binary mask if the given pixel is present or not
    """

    assert (depth.size(1) == 1)
    batch_size, _, N, M = depth.size()
    device = depth.device
    # Turn depth around. This also avoids problems with inplace operations
    depth = -depth.permute(0, 1, 3, 2)

    zero_one_row = torch.tensor([[0., 0., 0., 1.]])
    zero_one_row = zero_one_row.expand(batch_size, 1, 4).to(device)

    # add row to world mat
    cam_W = torch.cat((cam_W, zero_one_row), dim=1)

    # clean depth image for mask
    mask = (depth.abs() != float("Inf")).float()
    depth[depth == float("Inf")] = 0
    depth[depth == -1 * float("Inf")] = 0

    # 4d array to 2d array k=N*M
    d = depth.reshape(batch_size, 1, N * M)

    # create pixel location tensor
    px, py = torch.meshgrid([torch.arange(0, N), torch.arange(0, M)])
    px, py = px.to(device), py.to(device)

    p = torch.cat((
        px.expand(batch_size, 1, px.size(0), px.size(1)),
        (M - py).expand(batch_size, 1, py.size(0), py.size(1))
    ), dim=1)
    p = p.reshape(batch_size, 2, py.size(0) * py.size(1))
    p = (p.float() / M * 2)

    # create terms of mapping equation x = P^-1 * d*(qp - b)
    P = cam_K[:, :2, :2].float().to(device)
    q = cam_K[:, 2:3, 2:3].float().to(device)
    b = cam_K[:, :2, 2:3].expand(batch_size, 2, d.size(2)).to(device)
    Inv_P = torch.inverse(P).to(device)

    rightside = (p.float() * q.float() - b.float()) * d.float()
    x_xy = torch.bmm(Inv_P, rightside)

    # add depth and ones to location in world coord system
    x_world = torch.cat((x_xy, d, torch.ones_like(d)), dim=1)

    # derive loactoion in object coord via loc3d = W^-1 * x_world
    Inv_W = torch.inverse(cam_W)
    loc3d = torch.bmm(
        Inv_W.expand(batch_size, 4, 4),
        x_world
    ).reshape(batch_size, 4, N, M)

    loc3d = loc3d[:, :3].to(device)
    mask = mask.to(device)
    return loc3d, mask


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str)
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)
    parser.add_argument('-m', '--mesh_folder', type=str)

    args = parser.parse_args()
    # visualize_depth("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/tex_field/shape06082_rank00_pair248286")
    render_with_photoshape_views(args.proc, args.num_proc)
