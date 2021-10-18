from collections import defaultdict

import trimesh
from pathlib import Path
from tqdm import tqdm
import numpy as np
import torch

from util.misc import read_list


def get_face_neighbors(mesh):
    face_neighbors = [[-1, -1, -1] for _ in range(mesh.faces.shape[0])]
    edge_faces = defaultdict(list)
    for f_idx in range(mesh.faces.shape[0]):
        v_0, v_1, v_2 = mesh.faces[f_idx]
        edge_faces[frozenset({v_0, v_1})].append(f_idx)
        edge_faces[frozenset({v_1, v_2})].append(f_idx)
        edge_faces[frozenset({v_2, v_0})].append(f_idx)

    for f_idx in range(mesh.faces.shape[0]):
        v_0, v_1, v_2 = mesh.faces[f_idx]
        for neighbor_face in edge_faces[frozenset({v_0, v_1})] + edge_faces[frozenset({v_1, v_2})] + edge_faces[frozenset({v_2, v_0})]:
            if neighbor_face != f_idx:
                if neighbor_face in edge_faces[frozenset({v_0, v_1})]:
                    face_neighbors[f_idx][0] = neighbor_face
                elif neighbor_face in edge_faces[frozenset({v_1, v_2})]:
                    face_neighbors[f_idx][1] = neighbor_face
                elif neighbor_face in edge_faces[frozenset({v_2, v_0})]:
                    face_neighbors[f_idx][2] = neighbor_face

    # create padding - new faces and new vertices
    new_faces = []
    new_vertices = []
    for f_idx in range(mesh.faces.shape[0]):
        f_0, f_1, f_2 = face_neighbors[f_idx]
        vi_0, vi_1, vi_2 = mesh.faces[f_idx]
        v_0, v_1, v_2 = mesh.vertices[vi_0, :], mesh.vertices[vi_1, :], mesh.vertices[vi_2, :]
        if f_0 == -1:
            new_vertices.append(v_0 + v_1 - v_2)
            new_faces.append([vi_0, (len(mesh.vertices) - 1) + len(new_vertices), vi_1])
            face_neighbors[f_idx][0] = mesh.faces.shape[0] + len(new_faces) - 1
        if f_1 == -1:
            new_vertices.append(v_1 + v_2 - v_0)
            new_faces.append([vi_1, (len(mesh.vertices) - 1) + len(new_vertices), vi_2])
            face_neighbors[f_idx][1] = mesh.faces.shape[0] + len(new_faces) - 1
        if f_2 == -1:
            new_vertices.append(v_0 + v_2 - v_1)
            new_faces.append([vi_0, vi_2, (len(mesh.vertices) - 1) + len(new_vertices)])
            face_neighbors[f_idx][2] = mesh.faces.shape[0] + len(new_faces) - 1
    vertices = np.zeros((mesh.vertices.shape[0] + len(new_vertices), 3), dtype=np.float32)
    vertices[:mesh.vertices.shape[0], :] = mesh.vertices
    faces = np.zeros((mesh.faces.shape[0] + len(new_faces), 3), dtype=np.int64)
    faces[:mesh.faces.shape[0], :] = mesh.faces
    is_pad_vertex = np.zeros((mesh.vertices.shape[0] + len(new_vertices)), dtype=np.bool)
    is_pad_face = np.zeros((mesh.faces.shape[0] + len(new_faces)), dtype=np.bool)

    if len(new_vertices) > 0:
        vertices[mesh.vertices.shape[0]:, :] = np.array(new_vertices)
        faces[mesh.faces.shape[0]:, :] = np.array(new_faces)
        is_pad_vertex[mesh.vertices.shape[0]:] = True
        is_pad_face[mesh.faces.shape[0]:] = True

    return face_neighbors, vertices, faces, is_pad_vertex, is_pad_face


def cartesian_ordering(face_neighbors, faces, vertices):
    def get_least_index(f_idxs, f_num, dim=0):
        individual_mins = [vertices[faces[f_idx], dim].min() for f_idx in f_idxs]
        global_min = min(individual_mins)
        argmin_global = np.argmin(individual_mins)
        if dim < 2:
            next_level_f_idxs, next_level_f_nums = [], []
            for i in range(len(f_idxs)):
                if global_min == individual_mins[i]:
                    next_level_f_idxs.append(f_idxs[i])
                    next_level_f_nums.append(f_num[i])
            if len(next_level_f_idxs) == 1:
                return f_num[argmin_global]
            return get_least_index(next_level_f_idxs, next_level_f_nums, dim + 1)
        return f_num[argmin_global]

    for face_neighbor_idx, face_neighbor in enumerate(face_neighbors):
        least_idx = get_least_index(face_neighbor, list(range(3)), 0)
        reordered_face_neighbors = [face_neighbor[least_idx], face_neighbor[(least_idx + 1) % 3], face_neighbor[(least_idx + 2) % 3]]
        face_neighbors[face_neighbor_idx] = reordered_face_neighbors

    return face_neighbors


def append_self_face(face_neighbors):
    for face_neighbor_idx, face_neighbor in enumerate(face_neighbors):
        face_neighbors[face_neighbor_idx] = [face_neighbor_idx] + face_neighbors[face_neighbor_idx]
    return face_neighbors


def calculate_face_hierarchies(decimation_mesh):
    from scipy.spatial import KDTree
    pool_down_locations = []
    for i in range(1, len(decimation_mesh)):
        mesh_src = decimation_mesh[i - 1]
        mesh_tgt = decimation_mesh[i]
        num_samples = max(min(mesh_src.faces.shape[0] * 50 * i * i, 500000), mesh_src.faces.shape[0] * i * i)
        samples_src, face_idx_src = trimesh.sample.sample_surface_even(mesh_src, num_samples)
        samples_tgt, face_idx_tgt = trimesh.sample.sample_surface_even(mesh_tgt, num_samples)
        tree = KDTree(samples_tgt)
        distances, indices = tree.query(samples_src, k=1)
        face_map_src_tgt = np.zeros(mesh_src.faces.shape[0], np.int64)
        for face_idx in range(mesh_src.faces.shape[0]):
            # TODO: what happens when no point lands on face_idx_src ? At the moment, crashes.
            mask_src_faces = face_idx_src == face_idx
            uniques, unique_counts = np.unique(face_idx_tgt[indices[mask_src_faces]], return_counts=True)
            assert (uniques.shape[0] > 0 and unique_counts.shape[0] > 0), f"{uniques.shape[0]}, {unique_counts.shape[0]} unique NNs"
            face_map_src_tgt[face_idx] = uniques[np.argmax(unique_counts)]
        pool_down_locations.append(face_map_src_tgt)
    return pool_down_locations


def process_mesh(mesh_input_directory, output_processed_directory):
    already_exists = np.all([(output_processed_directory / f"{mesh_input_directory.name}_{i:03d}_000.pt").exists() for i in range(12)])
    if already_exists:
        return
    mesh_src_paths = [mesh_input_directory / f"inv_partial_texture_{i:03d}.obj" for i in range(12)]
    mesh_target_path = mesh_input_directory / "surface_texture.obj"
    decimated_paths = sorted([x for x in Path(mesh_target_path).parent.iterdir() if x.name.startswith('decimate') and x.name.endswith('.obj')])
    decimations = []
    mesh = trimesh.load(mesh_target_path, process=False, force='mesh')
    decimations.append(mesh)
    for dec_path in decimated_paths:
        decimations.append(trimesh.load(dec_path, process=False, force='mesh'))
    conv_data = []
    for d in decimations:
        face_neighbors, vertices, faces, is_pad_vertex, is_pad_face = get_face_neighbors(d)
        face_neighbors = cartesian_ordering(face_neighbors, faces, vertices)
        face_neighbors = append_self_face(face_neighbors)
        conv_data.append((face_neighbors, vertices, faces, is_pad_vertex, is_pad_face))
    pool_locations = calculate_face_hierarchies(decimations)
    face_colors_target = mesh.visual.face_colors
    for i, ms in enumerate(mesh_src_paths):
        in_mesh = trimesh.load(ms, process=False, force='mesh')
        face_colors_src = in_mesh.visual.face_colors
        torch.save({
            "input_positions": torch.from_numpy(in_mesh.triangles.mean(axis=1)).float(),
            "input_colors": torch.from_numpy(face_colors_src[:, :3]).float() / 255 - 0.5,
            "target_colors": torch.from_numpy(face_colors_target[:, :3]).float() / 255 - 0.5,
            "valid_input_colors": torch.from_numpy(face_colors_src[:, 3]).float() / 255,
            "valid_target_colors": torch.from_numpy(face_colors_target[:, 3]).float() / 255,
            "pool_locations": [torch.from_numpy(p).long() for p in pool_locations],
            "conv_data": [[torch.from_numpy(np.array(cd[0])).long(), torch.from_numpy(cd[1]).float(),
                          torch.from_numpy(cd[2]).long(), torch.from_numpy(cd[3]).bool(),
                          torch.from_numpy(cd[4]).bool()] for cd in conv_data]
        }, output_processed_directory / f"{mesh_input_directory.name}_{i:03d}_000.pt")
    return


def all_export(proc, n_proc):
    dataset = "SingleShape/CubeTexturePlane"
    split = "official"
    items = sorted(list(set(read_list("data/splits/" + dataset + f"/{split}/train.txt") + read_list("data/splits/" + dataset + f"/{split}/val.txt"))))
    print("Length of items: ", len(items))
    mesh_in_dirs = [Path("data", dataset, item) for item in items]
    mesh_in_dirs = [x for i, x in enumerate(mesh_in_dirs) if i % n_proc == proc]
    processed_out_dir = Path(f"data/{dataset}_FC_processed/")
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
