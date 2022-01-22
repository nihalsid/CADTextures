from datetime import datetime
from pathlib import Path

import pyrender
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from pyrender import RenderFlags
from PIL import Image
import logging
import json
from tqdm import tqdm
import random
import math

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


def create_raymond_lights():
    num_lights = 3
    angles = np.linspace(0, 2, num=num_lights + 1)[:num_lights].tolist()
    cphis = [np.pi * np.array(angles), np.pi * np.array([x + 1.0 / 3.0 for x in angles]), np.pi * np.array(angles), np.pi * np.array([0])]
    cthetas = [np.pi * np.array([1.0 / 3.0] * num_lights), np.pi * np.array([1.0 / 2.0] * num_lights), np.pi * np.array([2.0 / 3.0] * num_lights), np.pi * np.array([0])]
    intensities = [0.25, 0.25, 0.15, 10]
    nodes = []

    for phis, thetas, intensity in zip(cphis, cthetas, intensities):

        for phi, theta in zip(phis, thetas):
            xp = np.sin(theta) * np.cos(phi)
            zp = np.sin(theta) * np.sin(phi)
            yp = np.cos(theta)

            z = np.array([xp, yp, zp])
            z = z / np.linalg.norm(z)
            x = np.array([-z[1], z[0], 0.0])
            if np.linalg.norm(x) == 0:
                x = np.array([1.0, 0.0, 0.0])
            x = x / np.linalg.norm(x)
            y = np.cross(z, x)

            matrix = np.eye(4)
            matrix[:3, :3] = np.c_[x, y, z]
            nodes.append(pyrender.Node(
                light=pyrender.DirectionalLight(color=np.ones(3), intensity=intensity),
                matrix=matrix
            ))

    return nodes


def render_mesh(mesh, output_dir):
    output_dir.mkdir(exist_ok=True, parents=True)
    lights = create_raymond_lights()
    width = height = 2 ** 10
    r = pyrender.OffscreenRenderer(width, height)
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0, znear=0.001)
    camera_intrinsics = np.eye(4, dtype=np.float32)
    camera_intrinsics[0, 0] = camera_intrinsics[1, 1] = height / (2 * np.tan(camera.yfov / 2.0))
    camera_intrinsics[0, 2] = camera_intrinsics[1, 2] = height / 2

    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0], ambient_light=[0.2, 0.2, 0.2])
    for geo in mesh.geometry:
        geo = pyrender.Mesh.from_trimesh(mesh.geometry[geo])
        scene.add(geo)
    for n in lights:
        scene.add_node(n, scene.main_camera_node)
    # original angles: x (180, 120, -15), y (0, 360, 45), z 180
    for x_angle in range(225, 60, -45):
        for y_angle in range(0, 360, 45):
            camera_pose = np.eye(4)
            camera_pose[:3, :3] = Rotation.from_euler('y', y_angle, degrees=True).as_matrix() @ Rotation.from_euler('z', 180, degrees=True).as_matrix() @ Rotation.from_euler('x', x_angle, degrees=True).as_matrix()
            camera_translation = camera_pose[:3, :3] @ np.array([0, 0, 1.375])
            camera_pose[:3, 3] = camera_translation
            cam_node = scene.add(camera, pose=camera_pose)
            color_flat, depth = r.render(scene, flags=RenderFlags.FLAT | RenderFlags.SKIP_CULL_FACES)
            Image.fromarray(color_flat).save(output_dir / f"flat_{x_angle:03d}_{y_angle:03d}.jpg")
            np.savez_compressed(output_dir / f"camera_{x_angle:03d}_{y_angle:03d}.npz", depth=depth, cam_intrinsic=camera_intrinsics, cam_extrinsic=camera_pose)
            scene.remove_node(cam_node)


def create_mesh_for_view(mesh, image, depth, projection_matrix, camera_pose, output_path):
    vertex_colors = get_projected_colors(mesh, image, depth, projection_matrix, camera_pose)
    trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors, process=False).export(output_path)


def create_mesh_from_all_views(mesh, images, depths, projection_matrices, camera_poses, output_path):
    vertex_color_buffer = np.zeros((mesh.vertices.shape[0], 4), dtype=np.float32)
    for i in range(len(images)):
        vertex_color_buffer += get_projected_colors(mesh, images[i], depths[i], projection_matrices[i], camera_poses[i])
    counts = vertex_color_buffer[:, 3] / 255
    non_zero_counts = np.where(counts != 0)[0]
    vertex_color_buffer[non_zero_counts, :] = vertex_color_buffer[non_zero_counts, :] / np.repeat(counts[non_zero_counts].reshape((-1, 1)), 4, axis=1)
    trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_color_buffer, process=False).export(output_path)


def get_projected_colors(mesh, image, depth, projection_matrix, camera_pose):
    vertices = np.vstack([mesh.vertices.T, np.ones((1, mesh.vertices.shape[0]))])
    projection_matrix[0, 0] = -projection_matrix[0, 0]
    uv = projection_matrix @ np.linalg.inv(camera_pose) @ vertices
    uv[0, :] = uv[0, :] / uv[2, :]
    uv[1, :] = uv[1, :] / uv[2, :]
    uv_int = np.round(uv).astype(np.int32)
    uv_int_mask = np.logical_and(np.logical_and(uv_int[0, :] >= 0, uv_int[0, :] < depth.shape[1]),
                                 np.logical_and(uv_int[1, :] >= 0, uv_int[1, :] < depth.shape[0]))
    d0 = depth[uv_int[1, uv_int_mask], uv_int[0, uv_int_mask]]
    d1 = -uv[2, uv_int_mask]
    depth_diff = np.sqrt((d0 - d1) ** 2)
    uv_int_mask[np.where(uv_int_mask)[0]] = depth_diff < 0.05
    valid_colors = image[uv_int[1, uv_int_mask], uv_int[0, uv_int_mask], :]
    vertex_colors = np.zeros((mesh.vertices.shape[0], 4), dtype=np.float32)
    vertex_colors[uv_int_mask, :3] = valid_colors
    vertex_colors[uv_int_mask, 3] = 255
    return vertex_colors
    # image[uv_int[1, uv_int_mask], uv_int[0, uv_int_mask], :] = [255, 0, 0]
    # Image.fromarray(image).save("test.jpg")


def render_and_export(input_path, proc, num_proc):
    meshes = sorted([(x / "model_normalized.obj") for x in input_path.iterdir() if (x / "model_normalized.obj").exists()], key=lambda x: x.name)
    meshes = [x for i, x in enumerate(meshes) if i % num_proc == proc]
    # meshes = [Path('/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs/shape02332_rank01_pair15567/model_normalized.obj')]

    logger.info(f'Proc {proc + 1}/{num_proc} processing {len(meshes)}')
    for mesh in tqdm(meshes):
        (mesh.parent / "render").mkdir(exist_ok=True)
        mesh_geometry = trimesh.load(mesh, force='scene', process=False)
        transform_matrix = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        mesh_geometry.apply_transform(transform_matrix)
        bounds = mesh_geometry.bounds
        loc = (bounds[0] + bounds[1]) / 2
        mesh_geometry.apply_translation(-loc)
        mesh_geometry.apply_scale(1 / (bounds[1] - bounds[0]).max())
        # mesh_geometry.export(str(mesh.parent / "test.obj"), file_type='obj')
        render_mesh(mesh_geometry, mesh.parent / "render")


def project_and_export(render_path, mesh_path, proc, num_proc):
    quad_meshes = {x.name.split('_')[0]: x / "quad_24576_272.obj" for x in mesh_path.iterdir()}
    render_dirs = sorted([(x / "render") for x in render_path.iterdir() if (x / "render").exists()], key=lambda x: x.name)
    render_dirs = [x for i, x in enumerate(render_dirs) if i % num_proc == proc]
    # render_dirs = [Path('/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs/shape02332_rank01_pair15567/render')]
    logger.info(f'Proc {proc + 1}/{num_proc} processing {len(render_dirs)}')
    for render_directory in tqdm(render_dirs):
        if render_directory.parent.name.split('_')[0] in quad_meshes:
            mesh_geometry = trimesh.load(quad_meshes[render_directory.parent.name.split('_')[0]], process=False)
            flat_render_list = sorted([x for x in render_directory.iterdir() if x.name.startswith('flat_')])
            flat_renders, depths, projection_matrices, camera_poses = [], [], [], []
            for fr in flat_render_list:
                flat_render = np.array(Image.open(fr))
                rx, ry = fr.stem.split('_')[1:3]
                camera = np.load(f"{fr.parent}/camera_{rx}_{ry}.npz")
                flat_renders.append(flat_render)
                depths.append(camera['depth'])
                projection_matrices.append(camera['cam_intrinsic'])
                camera_poses.append(camera['cam_extrinsic'])
            create_mesh_from_all_views(mesh_geometry, flat_renders, depths, projection_matrices, camera_poses, render_directory.parent / f"vcf_model_normalized.obj")
        else:
            print(f"{str(render_directory)} doesn't have a corresponding shape")


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


def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
    return result


def render_with_photoshape_view(mesh, view, vc_mode):
    lights = create_raymond_lights()
    width = height = 1000
    spherical_camera = spherical_coord_to_cam(view['fov'], view['azimuth'], view['elevation'])
    camera_pose = np.linalg.inv(spherical_camera.view_mat())
    r = pyrender.OffscreenRenderer(width, height)
    camera = pyrender.PerspectiveCamera(yfov=np.pi * view['fov'] / 180, aspectRatio=1.0, znear=0.001)
    if vc_mode:
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.5, 0.5, 0.5])
        geo = pyrender.Mesh.from_trimesh(mesh)
        geo.primitives[0].material.metallicFactor = 0.6
        geo.primitives[0].material.roughnessFactor = 0.4
        scene.add(geo)
    else:
        scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0, 0.0], ambient_light=[0.75, 0.75, 0.75])
        for geo in mesh.geometry:
            geo = pyrender.Mesh.from_trimesh(mesh.geometry[geo])
            scene.add(geo)
    for n in lights:
        scene.add_node(n, scene.main_camera_node)
    scene.add(camera, pose=camera_pose)
    color_flat, depth = r.render(scene, flags=RenderFlags.FLAT | RenderFlags.SKIP_CULL_FACES | RenderFlags.RGBA)
    color, depth = r.render(scene, flags=RenderFlags.SKIP_CULL_FACES | RenderFlags.SHADOWS_ALL)
    mask = color_flat[:, :, 3]
    nzy, nzx = np.nonzero(mask)
    color_flat = expand2square(Image.fromarray(color_flat[nzy.min(): nzy.max(), nzx.min():nzx.max(), :3]), (255, 255, 255)).resize((width, height))
    color = expand2square(Image.fromarray(color[nzy.min(): nzy.max(), nzx.min():nzx.max(), :3]), (255, 255, 255)).resize((width, height))
    mask = expand2square(Image.fromarray(mask[nzy.min(): nzy.max(), nzx.min():nzx.max()]), (0,)).resize((width, height), resample=Image.NEAREST)
    return color_flat, color, mask


def render_with_photoshape_views(proc, num_proc, vc_mode=False, random_views=False):
    suffix = f'_rand' if random_views else ''
    prefix = f'vc_' if vc_mode else ''
    pairmeta_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/metadata/pairs.json")
    image_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/exemplars")
    mesh_path = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs")
    output_path_flat = Path(f"/cluster/gimli/ysiddiqui/CADTextures/Photoshape/exemplars_{prefix}simulated_nolight{suffix}")
    output_path_light = Path(f"/cluster/gimli/ysiddiqui/CADTextures/Photoshape/exemplars_{prefix}simulated_light{suffix}")
    output_path_mask = Path(f"/cluster/gimli/ysiddiqui/CADTextures/Photoshape/exemplars_{prefix}simulated_mask{suffix}")
    output_path_flat.mkdir(exist_ok=True)
    output_path_light.mkdir(exist_ok=True)
    output_path_mask.mkdir(exist_ok=True)
    views = load_pair_meta_views(image_path, pairmeta_path)
    view_keys = sorted(list(views.keys()))
    view_keys = [x for i, x in enumerate(view_keys) if i % num_proc == proc]
    for v in tqdm(view_keys):
        if (mesh_path / v / "vc_model_normalized.obj").exists():
            if vc_mode:
                mesh_geometry = trimesh.load(mesh_path / v / "vc_model_normalized.obj", force='mesh', process=False)
            else:
                mesh_geometry = trimesh.load(mesh_path / v / "model_normalized.obj", force='scene', process=False)
            transform_matrix = np.array([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
            mesh_geometry.apply_transform(transform_matrix)
            bounds = mesh_geometry.bounds
            loc = (bounds[0] + bounds[1]) / 2
            mesh_geometry.apply_translation(-loc)
            mesh_geometry.apply_scale(1 / (bounds[1] - bounds[0]).max())
            try:
                flat, light, mask = render_with_photoshape_view(mesh_geometry, views[v] if not random_views else get_random_views(1)[0], vc_mode)
                flat.save(output_path_flat / f"{v}.jpg")
                light.save(output_path_light / f"{v}.jpg")
                mask.save(output_path_mask / f"{v}.jpg")
            except Exception as err:
                print('Failed for', v, err)
        else:
            print(mesh_path / v / "model_normalized.obj", "doesn't exist")


def get_random_views(num_views):
    elevation_params = [1.407, 0.207, 0.785, 1.767]
    azimuth = random.sample(np.arange(0, 2 * math.pi).tolist(), num_views)
    elevation = np.random.uniform(low=elevation_params[2], high=elevation_params[3], size=num_views).tolist()
    return [{'azimuth': a, 'elevation': e, 'fov': 50} for a, e in zip(azimuth, elevation)]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str)
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)
    parser.add_argument('-m', '--mesh_folder', type=str)

    args = parser.parse_args()
    # render_with_photoshape_views(args.proc, args.num_proc, True, False)
    # project_and_export(Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs"), Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs-manifold-highres"), args.proc, args.num_proc)
