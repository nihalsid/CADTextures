from datetime import datetime
from pathlib import Path

import pyrender
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation
from pyrender import RenderFlags
from PIL import Image
import logging

# setup logging
from tqdm import tqdm

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
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

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
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))
    return nodes


def render_mesh(mesh, output_dir):
    output_dir.mkdir(exist_ok=True, parents=True)
    bounds = mesh.bounds
    loc = (bounds[0] + bounds[1]) / 2
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
            camera_translation = camera_pose[:3, :3] @ np.array([0, 0, 1.025]) + loc
            # camera_translation = camera_pose[:3, :3] @ np.array([0, 0, 3.525]) + loc
            camera_pose[:3, 3] = camera_translation
            cam_node = scene.add(camera, pose=camera_pose)
            color_flat, depth = r.render(scene, flags=RenderFlags.FLAT)
            color, _ = r.render(scene)
            Image.fromarray(color_flat).save(output_dir / f"flat_{x_angle:03d}_{y_angle:03d}.jpg")
            Image.fromarray(color).save(output_dir / f"shade_{x_angle:03d}_{y_angle:03d}.jpg")
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
    uv_int_mask[np.where(uv_int_mask)[0]] = depth_diff < 0.01
    valid_colors = image[uv_int[1, uv_int_mask], uv_int[0, uv_int_mask], :]
    vertex_colors = np.zeros((mesh.vertices.shape[0], 4), dtype=np.float32)
    vertex_colors[uv_int_mask, :3] = valid_colors
    vertex_colors[uv_int_mask, 3] = 255
    return vertex_colors
    # image[uv_int[1, uv_int_mask], uv_int[0, uv_int_mask], :] = [255, 0, 0]
    # Image.fromarray(image).save("test.jpg")


def render_and_export(input_path, export_path, proc, num_proc):
    export_path.mkdir(exist_ok=True, parents=True)
    meshes = sorted([(x / "models" / "model_normalized.obj") for x in input_path.iterdir() if (x / "models" / "model_normalized.obj").exists()], key=lambda x: x.name)
    # meshes = sorted([(x / "model_normalized.obj") for x in input_path.iterdir() if (x / "model_normalized.obj").exists()], key=lambda x: x.name)
    meshes = [x for i, x in enumerate(meshes) if i % num_proc == proc]
    # splitlist = Path("data/splits/SingleShape/CubeTexturesForGraph/overfit_050/train.txt").read_text().splitlines()
    # meshes = [x for x in meshes if x.parent.name in splitlist]
    # meshes = [x for x in meshes if x.parents[1].name in splitlist]
    logger.info(f'Proc {proc + 1}/{num_proc} processing {len(meshes)}')
    for mesh in tqdm(meshes):
        parent_dir = mesh.parents[1].name
        # parent_dir = mesh.parents[0].name
        (export_path / parent_dir / "render").mkdir(exist_ok=True)
        mesh_geometry = trimesh.load(mesh, force='scene')
        render_mesh(mesh_geometry, export_path / parent_dir / "render")


def project_and_export(input_path, proc, num_proc):
    meshes = sorted([(x / "model_normalized.obj") for x in input_path.iterdir() if (x / "model_normalized.obj").exists()], key=lambda x: x.name)
    meshes = [x for i, x in enumerate(meshes) if i % num_proc == proc]
    # splitlist = Path("data/splits/SingleShape/CubeTexturesForGraph/overfit_050/train.txt").read_text().splitlines()
    # meshes = [x for x in meshes if x.parent.name in splitlist]
    logger.info(f'Proc {proc + 1}/{num_proc} processing {len(meshes)}')
    for mesh in tqdm(meshes):
        mesh_geometry = trimesh.load(mesh, force='mesh', process=False)
        flat_render_list = sorted([x for x in (mesh.parent / "render").iterdir() if x.name.startswith('flat_')])
        flat_renders, depths, projection_matrices, camera_poses = [], [], [], []
        for fr in flat_render_list:
            flat_render = np.array(Image.open(fr))
            rx, ry = fr.stem.split('_')[1:3]
            camera = np.load(f"{fr.parent}/camera_{rx}_{ry}.npz")
            flat_renders.append(flat_render)
            depths.append(camera['depth'])
            projection_matrices.append(camera['cam_intrinsic'])
            camera_poses.append(camera['cam_extrinsic'])
            create_mesh_for_view(mesh_geometry, flat_render, camera['depth'], camera['cam_intrinsic'], camera['cam_extrinsic'], mesh.parent / f"model_normalized_input_{rx}_{ry}.obj")
        create_mesh_from_all_views(mesh_geometry, flat_renders, depths, projection_matrices, camera_poses, mesh.parent / f"model_normalized.obj")


if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str)
    parser.add_argument('-o', '--output_folder', type=str)
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)

    args = parser.parse_args()
    render_and_export(Path(args.input_folder), Path(args.output_folder), args.proc, args.num_proc)
    project_and_export(Path(args.output_folder), args.proc, args.num_proc)
