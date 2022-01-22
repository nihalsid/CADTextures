import numpy as np
import torch
import trimesh
from scipy.sparse import coo_matrix
import json


def laplace_weights(face_centers, neighbors, equal_weight=False):
    neighbors_list = []
    for i in range(neighbors.shape[0]):
        neighbors_list.append(neighbors[i][neighbors[i] < neighbors.shape[0]].tolist())

    col = np.concatenate(neighbors_list)
    row = np.concatenate([[i] * len(n) for i, n in enumerate(neighbors_list)])

    if equal_weight:
        # equal weights for each neighbor
        data = np.concatenate([[1.0 / len(n)] * len(n) for n in neighbors_list])
    else:
        # umbrella weights, distance-weighted
        # use dot product of ones to replace array.sum(axis=1)
        ones = np.ones(3)
        norms = [
            1.0 / np.maximum(1e-6, np.sqrt(np.dot((face_centers[i] - face_centers[n]) ** 2, ones)))
            for i, n in enumerate(neighbors_list)]
        data = np.concatenate([i / i.sum() for i in norms])

    matrix = coo_matrix((data, (row, col)), shape=[len(face_centers)] * 2)
    return matrix


def fundamental_forms(mesh, neighbors):
    # first form = lengths of sides
    quad = mesh.triangles
    side_0 = np.linalg.norm(quad[:, 0, :] - quad[:, 1, :], axis=1).reshape(-1, 1)
    side_1 = np.linalg.norm(quad[:, 1, :] - quad[:, 2, :], axis=1).reshape(-1, 1)
    side_2 = np.linalg.norm(quad[:, 2, :] - quad[:, 3, :], axis=1).reshape(-1, 1)
    side_3 = np.linalg.norm(quad[:, 3, :] - quad[:, 0, :], axis=1).reshape(-1, 1)
    # second form = angles
    neighbor_similarity = []
    normals = np.array(mesh.face_normals)
    for i in range(neighbors.shape[0]):
        neighbor_similarity_i = []
        normal_i = normals[i]
        for j in range(neighbors.shape[1]):
            if neighbors[i, j] < neighbors.shape[0]:
                normal_j = normals[neighbors[i, j]]
            else:
                normal_j = normal_i
            dot = np.sum(normal_i * normal_j, axis=0).clip(-1, 1)
            angle = np.pi - np.arccos(dot)
            neighbor_similarity_i.append(angle)
        neighbor_similarity.append(neighbor_similarity_i)
    neighbor_similarity = np.array(neighbor_similarity)
    return np.concatenate([side_0, side_1, side_2, side_3], axis=1), neighbor_similarity


def calculate_curvature(mesh, num_pads):
    v_gaussian_curvature = trimesh.curvature.discrete_gaussian_curvature_measure(mesh, mesh.vertices, 0)
    v_mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh, mesh.vertices, 0.015)
    v_gaussian_curvature[v_gaussian_curvature.shape[0] - num_pads:] = v_gaussian_curvature.mean()
    v_mean_curvature[v_mean_curvature.shape[0] - num_pads:] = v_mean_curvature.mean()
    f_gaussian_curvature = v_gaussian_curvature[mesh.faces[:, 0]]
    f_mean_curvature = v_mean_curvature[mesh.faces[:, 0]]
    for i in range(1, 4):
        f_gaussian_curvature += v_gaussian_curvature[mesh.faces[:, i]]
        f_mean_curvature += v_mean_curvature[mesh.faces[:, i]]
    f_gaussian_curvature /= 4
    f_mean_curvature /= 4
    return f_gaussian_curvature, f_mean_curvature


def add_features(mesh_path, pt_path):
    mesh = trimesh.load(mesh_path, process=False)
    face_centers = mesh.triangles.mean(1)
    vflist = [x.split(' ')[0] for x in mesh_path.read_text().splitlines() if x.split(' ')[0] in ['v', 'f', '']]
    num_pads = 0
    if '' in vflist:
        blankpos = len(vflist) - 1 - vflist[::-1].index('')
        num_pads = vflist[blankpos + 1:].index('f')
    pt_arxiv = torch.load(pt_path)
    neighborhood = pt_arxiv['conv_data'][0][0].long().numpy()[:, 1:]
    laplacian = torch.from_numpy(laplace_weights(face_centers, neighborhood) * face_centers).float()
    ff1, ff2 = fundamental_forms(mesh, neighborhood)
    gcurv, mcurv = calculate_curvature(mesh, num_pads)
    pt_arxiv['input_laplacian'] = laplacian
    pt_arxiv['input_ff1'] = torch.from_numpy(ff1).float()
    pt_arxiv['input_ff2'] = torch.from_numpy(ff2).float()
    pt_arxiv['input_gcurv'] = torch.from_numpy(gcurv).float()
    pt_arxiv['input_mcurv'] = torch.from_numpy(mcurv).float()
    pt_arxiv['vertices'] = torch.from_numpy(mesh.vertices).float()
    pt_arxiv['indices'] = torch.from_numpy(mesh.faces).int()
    torch.save(pt_arxiv, pt_path)


def add_multilevel_positions(mesh_input_directory, pt_path):

    def get_selections(mesh_dir):
        selections = json.loads((mesh_dir / "selection.json").read_text())
        selections = {int(k): v for k, v in selections.items()}
        selections[24] = 1
        faces = sorted(list(selections.keys()), key=lambda x: int(x), reverse=True)
        qmeshes = [mesh_dir / f"quad_{int(f):05d}_{selections[f]:03d}.obj" for f in faces]
        return selections, qmeshes

    pt_arxiv = torch.load(pt_path)
    mesh_paths = get_selections(mesh_input_directory)[1]
    position_data = []
    for dec_path in mesh_paths:
        mesh = trimesh.load(dec_path, process=False)
        position_data.append(torch.from_numpy(np.array(mesh.triangles.mean(axis=1))).float())

    pt_arxiv['pos_data'] = position_data
    torch.save(pt_arxiv, pt_path)


def add_vertices_indices(mesh_path, pt_path):
    mesh = trimesh.load(mesh_path, process=False)
    pt_arxiv = torch.load(pt_path)
    pt_arxiv['vertices'] = torch.from_numpy(mesh.vertices).float()
    pt_arxiv['indices'] = torch.from_numpy(mesh.faces).int()
    pt_arxiv['normals'] = torch.from_numpy(np.array(mesh.vertex_normals)).float()
    torch.save(pt_arxiv, pt_path)


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    from tqdm import tqdm

    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)

    mesh_root = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/shapenet-chairs-manifold-highres")
    pt_root = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/shapenet-chairs-manifold-highres-part_processed_color/")
    data_root = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/shapenet-chairs-manifold-highres")

    args = parser.parse_args()
    files = sorted([x for x in pt_root.iterdir()])
    files = [x for i, x in enumerate(files) if i % args.num_proc == args.proc]

    for pt in tqdm(files):
        # add_features(mesh_root / pt.name.split('.')[0] / "model_normalized.obj", pt)
        # add_multilevel_positions(data_root / pt.name.split('.')[0], pt)
        add_vertices_indices(mesh_root / pt.name.split('.')[0] / "model_normalized.obj", pt)
