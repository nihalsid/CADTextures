from pathlib import Path
import trimesh
from argparse import ArgumentParser
import subprocess
import numpy as np
import marching_cubes as mc
import os

highres_dim = 64
padding_highres = 2
highres_voxel_size = 1 / (highres_dim - 2 * padding_highres)

print(f"VoxelRes: {highres_voxel_size}")

sdf_gen_highres_cmd = lambda in_filepath, out_filepath: f"data_processing/sdf_gen/bin/sdf_gen {in_filepath} {out_filepath} {highres_voxel_size} {padding_highres}"


def visualize_highres(df_path):
    df = np.load(str(df_path)+".npz")["arr"]
    vertices, triangles = mc.marching_cubes(df, highres_voxel_size * 0.75)
    mc.export_obj(vertices, triangles, str(df_path) + ".obj")


def export_distance_field(mesh_path, output_path, visualize=False):
    output_path.parents[0].mkdir(exist_ok=True, parents=True)
    failure_hr = subprocess.call(sdf_gen_highres_cmd(str(mesh_path), str(output_path)), shell=True)
    os.remove(str(output_path) + "_if.npy")
    df_highres = np.load(str(output_path)+".npy")
    print(df_highres.shape)
    df_highres_padded = df_highres
    df_highres_padded[np.abs(df_highres_padded) > 3 * highres_voxel_size] = 3 * highres_voxel_size
    np.savez_compressed(str(output_path), arr=df_highres_padded)
    os.remove(str(output_path) + ".npy")
    if visualize:
        visualize_highres(output_path)


if __name__ == '__main__':
    base_dir = Path(os.path.realpath(__file__)).parents[0].parents[0].parents[0]
    parser = ArgumentParser()
    parser.add_argument("--dataset", type=str, default='')
    parser.add_argument("--model_id", type=str, default='')
    args = parser.parse_args()
    dataset_0, dataset_1 = args.dataset.split('/')
    shape_path = Path(base_dir, f"data/{dataset_0}-model/{dataset_1}", args.model_id, "normalized_model.obj")
    output_path = Path(base_dir, f"data/{dataset_0}/{dataset_1}", args.model_id, f'shape_df')
    mesh = trimesh.load(shape_path, force='mesh', process=False)
    if type(mesh) != trimesh.Trimesh:
        mesh = mesh.dump().sum()
    bbox = mesh.bounding_box.bounds
    loc = (bbox[0] + bbox[1])/2
    mesh.apply_translation(-loc)
    scale = (bbox[1] - bbox[0]).max()
    mesh.apply_scale(1 / scale)
    obj_path = Path("/tmp/", args.model_id + ".obj")
    mesh.export(obj_path)
    export_distance_field(obj_path, output_path, visualize=True)
    os.remove(obj_path)
