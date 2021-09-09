import random
import shutil

import trimesh
from pathlib import Path
import numpy as np
import logging
from datetime import datetime
from tqdm import tqdm

# setup logging
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

SUBDIVISION_SIZE = 0.15


def subdivide_and_vertexcolor_mesh(mesh_inpath, mesh_outpath):
    logger.debug(f'Processing {mesh_inpath}')
    mesh = trimesh.load(mesh_inpath, force='scene')
    for geo_key in mesh.geometry.keys():
        mesh.geometry[geo_key] = mesh.geometry[geo_key].subdivide_to_size(SUBDIVISION_SIZE)
    replace_with_vertex_colors(mesh)
    mesh.export(str(mesh_outpath))
    logger.debug(f'Done with {mesh_inpath.name}')


def replace_with_vertex_colors(mesh):
    for geo_idx, geo_key in enumerate(mesh.geometry.keys()):
        if hasattr(mesh.geometry[geo_key].visual, 'uv') and mesh.geometry[geo_key].visual.uv is not None and mesh.geometry[geo_key].visual.uv.shape[0] > 0 and mesh.geometry[geo_key].visual.material.image is not None:
            colors = trimesh.visual.uv_to_color(mesh.geometry[geo_key].visual.uv, mesh.geometry[geo_key].visual.material.image)
        elif hasattr(mesh.geometry[geo_key].visual, 'uv') and hasattr(mesh.geometry[geo_key].visual, 'material'):
            colors = np.tile(mesh.geometry[geo_key].visual.material.diffuse.reshape((1, -1)), (mesh.geometry[geo_key].vertices.shape[0], 1))
        elif hasattr(mesh.geometry[geo_key].visual, 'vertex_colors'):
            colors = mesh.geometry[geo_key].visual.vertex_colors
        else:
            logger.error("Unknown material specification for color transfer")
            raise NotImplementedError
        mesh.geometry[geo_key] = trimesh.Trimesh(vertices=mesh.geometry[geo_key].vertices, faces=mesh.geometry[geo_key].faces, vertex_colors=colors)


def subdivide_and_vertexcolor_meshes(input_folder, target_folder, process_id, total_procs):
    target_folder.mkdir(exist_ok=True, parents=True)
    meshes = sorted([(x / "models" / "model_normalized.obj") for x in input_folder.iterdir() if (x / "models" / "model_normalized.obj").exists()], key=lambda x: x.name)
    meshes = [x for i, x in enumerate(meshes) if i % total_procs == process_id]
    logger.info(f'Proc {process_id + 1}/{total_procs} processing {len(meshes)}')
    for mesh in tqdm(meshes):
        (target_folder / mesh.parents[1].name).mkdir(exist_ok=True)
        subdivide_and_vertexcolor_mesh(mesh, (target_folder / mesh.parents[1].name / "model_normalized.obj"))


def create_random_fraction_color_input(input_folder):
    meshes = sorted([(x / "model_normalized.obj") for x in input_folder.iterdir() if (x / "model_normalized.obj").exists()], key=lambda x: x.name)
    for m in meshes:
        mesh = trimesh.load(m, force='mesh', process=False)
        colors = mesh.visual.vertex_colors
        indices = random.sample(range(colors.shape[0]), int(colors.shape[0] * 0.4))
        colors[indices, :] = 0
        mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=colors, process=False)
        mesh.export(m.parent / "model_normalized_input.obj")


def get_list_of_simple_meshes(output_dir):
    simple_meshes = []
    for x in tqdm(output_dir.iterdir()):
        mesh = trimesh.load(x / "model_normalized.obj", force='mesh', process=False)
        if mesh.vertices.shape[0] < 1200 and mesh.faces.shape[0] < 3000:
            simple_meshes.append(x.name)
    Path("simple_meshes.txt").write_text('\n'.join(simple_meshes))


def copy_simple_meshes(output_dir, copy_dir):
    for x in Path("simple_meshes.txt").read_text().splitlines():
        shutil.copytree(output_dir / x, copy_dir / x)


def uniform_color_mesh():
    root = Path("data/ShapeNetV2/DatasetCube/")
    mesh = trimesh.load(root / "00/model_normalized.obj")

    for i in range(8):

        color = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230), (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255), (170, 110, 40), (255, 250, 200),
                 (128, 0, 0), (170, 255, 195), (128, 128, 0), (255, 215, 180), (0, 0, 128), (128, 128, 128), (255, 255, 255), (0, 0, 0)]
        # random.choice(color)
        selected_faces_idx = random.sample(list(range(mesh.faces.shape[0])), 20)
        vertex_indices = np.vstack([mesh.faces[selected_faces_idx, 0].reshape(-1, 1), mesh.faces[selected_faces_idx, 1].reshape(-1, 1), mesh.faces[selected_faces_idx, 2].reshape(-1, 1)]).flatten()
        vertex_colors = np.hstack([np.ones_like(mesh.vertices) * np.array(color[i]), 255 * np.ones((mesh.vertices.shape[0], 1))])
        (root / f"{i + 1:02d}").mkdir(exist_ok=True)
        trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors, process=False).export(root / f"{i + 1:02d}" / "model_normalized.obj")
        vertex_colors[vertex_indices, :] = 0
        trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors, process=False).export(root / f"{i + 1:02d}" / "model_normalized_input_150_180.obj")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str)
    parser.add_argument('-o', '--output_folder', type=str)
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)

    args = parser.parse_args()
    subdivide_and_vertexcolor_meshes(Path(args.input_folder), Path(args.output_folder), args.proc, args.num_proc)
    # create_random_fraction_color_input(Path(args.input_folder))
    # get_list_of_simple_meshes(Path(args.output_folder))
    # copy_simple_meshes(Path(args.output_folder), Path("data/ShapeNetV2/DatasetSimple_256"))
    # uniform_color_mesh()
