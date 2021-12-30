import json
from collections import OrderedDict
from pathlib import Path
import torch

from scipy.spatial import KDTree
import trimesh
import numpy as np
from tqdm import tqdm

from data_processing.add_mesh_features import laplace_weights, fundamental_forms, calculate_curvature
from data_processing.process_mesh_for_conv import quadface_8_neighbors, cartesian_ordering, append_self_face


def get_pool_down_map(mesh_src, mesh_tgt):
    num_samples = 100000
    samples_src, face_idx_src = trimesh.sample.sample_surface_even(mesh_src, num_samples)
    samples_src = np.concatenate([samples_src, mesh_src.triangles.mean(1)], axis=0)
    face_idx_src = np.concatenate([face_idx_src, np.array(list(range(mesh_src.faces.shape[0])))], axis=0)
    samples_tgt, face_idx_tgt = trimesh.sample.sample_surface_even(mesh_tgt, num_samples)
    tree = KDTree(samples_tgt)
    distances, indices = tree.query(samples_src, k=1)
    face_map_src_tgt = np.zeros(mesh_src.faces.shape[0], np.int64)
    for face_idx in range(mesh_src.faces.shape[0]):
        mask_src_faces = face_idx_src == face_idx
        uniques, unique_counts = np.unique(face_idx_tgt[indices[mask_src_faces]], return_counts=True)
        assert (uniques.shape[0] > 0 and unique_counts.shape[0] > 0), f"{uniques.shape[0]}, {unique_counts.shape[0]}, {mask_src_faces.sum()} unique NNs"
        face_map_src_tgt[face_idx] = uniques[np.argmax(unique_counts)]
    return face_map_src_tgt


def get_selections(mesh_dir):
    selections = json.loads((mesh_dir / "selection.json").read_text())
    selections = {int(k): v for k, v in selections.items()}
    selections[24] = 1
    faces = sorted(list(selections.keys()), key=lambda x: int(x), reverse=True)
    qmeshes = [mesh_dir / f"quad_{int(f):05d}_{selections[f]:03d}.obj" for f in faces]
    return selections, qmeshes


def calculate_face_hierarchies(mesh_dir):
    selections = get_selections(mesh_dir)[0]
    face_N_R = OrderedDict({
        (24576, 6144): [272],
        (6144, 1536): [96, 88, 80, 72, 64],
        (1536, 384): [64, 56, 48, 40, 32],
        (384, 96): [32, 24, 16],
        (96, 24): [16, 8],
        (24, 1): [1],
    })
    pool_down_pairs = list(face_N_R.keys())
    pool_down_locations = []
    for i in range(len(pool_down_pairs) - 1):
        src_N = pool_down_pairs[i][0]
        tgt_N = pool_down_pairs[i][1]
        src_mesh_res = selections[src_N]
        tgt_mesh_res = selections[tgt_N]
        src_tgt_res = face_N_R[pool_down_pairs[i]]
        tgt_ntgt_res = face_N_R[pool_down_pairs[i + 1]]
        R = src_tgt_res[src_tgt_res.index(src_mesh_res):] + tgt_ntgt_res[1:tgt_ntgt_res.index(tgt_mesh_res) + 1] + [tgt_mesh_res]
        N = [src_N] * (len(R) - 1) + [tgt_N]
        N_R_ = list(zip(N, R))
        N_R = [N_R_[0]]
        # N_R = [N_R[0], N_R[-1]]
        for n, r in N_R_[1:-1]:
            try:
                mesh = trimesh.load(mesh_dir / f"quad_{n:05d}_{r:03d}.obj", process=False)
                if np.isnan(np.sum(mesh.vertices)):
                    raise ArithmeticError
                N_R.append((n, r))
            except Exception as err:
                print(f"Nan Mesh found {mesh_dir}: {n}, {r}; {str(err)}")
        N_R.append(N_R_[-1])
        pool_down_maps = []
        for j in range(len(N_R) - 1):
            mesh_src = trimesh.load(mesh_dir / f"quad_{N_R[j][0]:05d}_{N_R[j][1]:03d}.obj", process=False)
            mesh_tgt = trimesh.load(mesh_dir / f"quad_{N_R[j + 1][0]:05d}_{N_R[j + 1][1]:03d}.obj", process=False)
            pool_down_maps.append(get_pool_down_map(mesh_src, mesh_tgt))

        face_map_src_tgt = pool_down_maps[0]
        for pool_down_map in pool_down_maps[1:]:
            face_map_src_tgt = pool_down_map[face_map_src_tgt]

        pool_down_locations.append(face_map_src_tgt)
    return pool_down_locations


def process_mesh(mesh_input_directory, output_processed_directory):
    mesh_paths = get_selections(mesh_input_directory)[1]
    decimations = []
    mesh = trimesh.load(mesh_paths[0], process=False)
    decimations.append(mesh)
    for dec_path in mesh_paths[1:]:
        decimations.append(trimesh.load(dec_path, process=False))
    conv_data = []
    for d in decimations:
        face_neighbors, vertices, faces, is_pad_vertex, is_pad_face = quadface_8_neighbors(d)
        face_neighbors = cartesian_ordering(face_neighbors, faces, vertices)
        face_neighbors = append_self_face(face_neighbors)
        conv_data.append((face_neighbors, vertices, faces, is_pad_vertex, is_pad_face))
    pool_locations = calculate_face_hierarchies(mesh_input_directory)
    face_centers = np.array(mesh.triangles.mean(axis=1))
    neighborhood = np.array(conv_data[0][0])[:, 1:]
    vflist = [x.split(' ')[0] for x in mesh_paths[0].read_text().splitlines() if x.split(' ')[0] in ['v', 'f', '']]
    num_pads = 0
    if '' in vflist:
        blankpos = len(vflist) - 1 - vflist[::-1].index('')
        num_pads = vflist[blankpos + 1:].index('f')
    laplacian = laplace_weights(face_centers, neighborhood) * face_centers
    ff1, ff2 = fundamental_forms(mesh, neighborhood)
    gcurv, mcurv = calculate_curvature(mesh, num_pads)
    torch.save({
        "input_positions": torch.from_numpy(face_centers).float(),
        "input_normals": torch.from_numpy(np.array(mesh.face_normals)).float(),
        "input_laplacian": torch.from_numpy(laplacian).float(),
        "input_ff1": torch.from_numpy(ff1).float(),
        "input_ff2": torch.from_numpy(ff2).float(),
        "input_gcurv": torch.from_numpy(gcurv).float(),
        "input_mcurv": torch.from_numpy(mcurv).float(),
        "target_colors": torch.zeros([mesh.face_normals.shape[0], 3]).float() / 255 - 0.5,
        "pool_locations": [torch.from_numpy(p).long() for p in pool_locations],
        "conv_data": [[torch.from_numpy(np.array(cd[0])).long(), torch.from_numpy(cd[1]).float(),
                      torch.from_numpy(cd[2]).long(), torch.from_numpy(cd[3]).bool(),
                      torch.from_numpy(cd[4]).bool()] for cd in conv_data]
    }, output_processed_directory / f"{mesh_input_directory.name}.pt")
    return


def all_export(proc, n_proc):
    dataset = "Photoshape-model/shapenet-chairs-manifold-highres"
    items = sorted([x.name for x in (Path("data") / dataset).iterdir()])
    print("Length of items: ", len(items))
    mesh_in_dirs = [Path("data", dataset, item) for item in items]
    mesh_in_dirs = [x for i, x in enumerate(mesh_in_dirs) if i % n_proc == proc]
    processed_out_dir = Path(f"data/Photoshape/shapenet-chairs-manifold-highres_processed/")
    processed_out_dir.mkdir(exist_ok=True, parents=True)
    for m in tqdm(mesh_in_dirs):
        process_mesh(m, processed_out_dir)


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)

    args = parser.parse_args()
    all_export(args.proc, args.num_proc)
