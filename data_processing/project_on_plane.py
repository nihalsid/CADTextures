import trimesh
from pathlib import Path
from PIL import Image
import numpy as np
import shutil
import os
from tqdm import tqdm

plane = trimesh.load("data/SingleShape/CubeTexturePlane/base/128_128.obj", process=False, force='mesh')
vertices = np.array(plane.vertices)
sort_indices = np.lexsort((vertices[:, 0], vertices[:, 1], vertices[:, 2]))


def project_on_plane(sample_input_dir, image_name, mask_name, output_path):
    image = np.asarray(Image.open(sample_input_dir + f'/{image_name}.png').resize((128, 128))).copy()
    normals = np.asarray(Image.open(sample_input_dir + f'/{mask_name}.png').resize((128, 128), resample=Image.NEAREST))
    mask = np.logical_and(np.logical_and(normals[:, :, 0] == 0, normals[:, :, 1] == 0), normals[:, :, 2] == 0)
    image[mask, :] = 0
    vertex_colors = np.zeros((vertices.shape[0], 4), dtype=np.uint8)
    sort_index_i = 0
    for i in range(127, -1, -1):
        for j in range(128):
            vertex_colors[sort_indices[sort_index_i], :3] = image[i, j, :]
            vertex_colors[sort_indices[sort_index_i], 3] = (1 - mask[i, j]) * 255
            sort_index_i += 1
        sort_index_i += 1
    trimesh.Trimesh(vertices=vertices, vertex_colors=vertex_colors, faces=plane.faces, process=False).export(output_path)


def process_sample(sample_input_dir, output_dir):
    images = ["surface_texture",] + list(f"inv_partial_texture_{i:03d}" for i in range(12))
    masks = ["surface_normals", ] + list(f"inv_partial_mask_{i:03d}" for i in range(12))
    output_paths = [output_dir + "/" + im + ".obj" for im in images]
    for i in range(len(images)):
        project_on_plane(sample_input_dir, images[i], masks[i], output_paths[i])

    decimations = ["064_064", "032_032", "016_016", "008_008", "004_004"]

    for i, d in enumerate(decimations):
        shutil.copyfile(f"data/SingleShape/CubeTexturePlane/base/{d}.obj", output_dir + f"/decimate_{i + 1}.obj")
    os.remove(output_dir + "/material.mtl")


def main(input_folder, output_folder, num_proc, proc):
    samples = os.listdir(input_folder)
    samples = [x for i, x in enumerate(samples) if i % num_proc == proc]
    for sample in tqdm(samples):
        os.makedirs(output_folder + "/" + sample, exist_ok=True)
        process_sample(input_folder + "/" + sample, output_folder + "/" + sample)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)

    args = parser.parse_args()
    main("data/SingleShape/CubeTextures/", "data/SingleShape/CubeTexturePlane/", args.num_proc, args.proc)