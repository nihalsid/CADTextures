import json
import shutil
import numpy as np
from pathlib import Path

import torch_scatter
import trimesh
from tqdm import tqdm
import torch

mesh_dir = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/shapenet-chairs-manifold-highres")
annotation_path = Path('/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/part-net/annotations')
level_0_path = Path('/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/part-net/Chair-level-1.txt')
aligned_annotation_path = Path('/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/part-net/aligned_annotations')
processed_shapes_path = Path('/cluster/gimli/ysiddiqui/CADTextures/Photoshape/shapenet-chairs-manifold-highres_processed')
processed_shapes_pruned_path = Path('/cluster/gimli/ysiddiqui/CADTextures/Photoshape/shapenet-chairs-manifold-highres-part_processed')
shape_meta_path = Path('/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/metadata/shapes.json')


def merge_annotation_map():
    map_0 = json.loads(Path("/home/nihalsid/Downloads/Chair.train.json").read_text())
    map_1 = json.loads(Path("/home/nihalsid/Downloads/Chair.val.json").read_text())
    map_2 = json.loads(Path("/home/nihalsid/Downloads/Chair.test.json").read_text())
    map_0.extend(map_1)
    map_0.extend(map_2)
    map_dict = {}
    for x in map_0:
        map_dict[x['model_id']] = x
    Path("/home/nihalsid/Downloads/Chair.json").write_text(json.dumps(map_dict))


def copy_annotations():
    model_to_annot = json.loads(Path('/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/part-net/Chair.json').read_text())
    part_net_path = Path('/cluster/pegasus/abokhovkin/datasets/PartNet/')
    destination_annot_path = annotation_path
    destination_annot_path.mkdir(exist_ok=True)
    for model_id in tqdm(model_to_annot.keys()):
        # (destination_annot_path / model_id).mkdir()
        shutil.copyfile(part_net_path / model_to_annot[model_id]['anno_id'] / "point_sample" / 'sample-points-all-pts-nor-rgba-10000.txt',
                        destination_annot_path / model_id / 'sample-points-all-pts-nor-rgba-10000.txt')
        shutil.copyfile(part_net_path / model_to_annot[model_id]['anno_id'] / "point_sample" / 'sample-points-all-label-10000.txt',
                        destination_annot_path / model_id / 'sample-points-all-label-10000.txt')
        shutil.copyfile(part_net_path / model_to_annot[model_id]['anno_id'] / "point_sample" / "pts-10000.txt",
                        destination_annot_path / model_id / "pts-10000.txt")
        shutil.copyfile(part_net_path / model_to_annot[model_id]['anno_id'] / "point_sample" / "label-10000.txt",
                        destination_annot_path / model_id / "label-10000.txt")
        shutil.copyfile(part_net_path / model_to_annot[model_id]['anno_id'] / "result_after_merging.json",
                        destination_annot_path / model_id / "result_after_merging.json")
        shutil.copyfile(part_net_path / model_to_annot[model_id]['anno_id'] / "result.json",
                        destination_annot_path / model_id / "result.json")


def copy_existing_data():
    shape_meta = json.loads(shape_meta_path.read_text())
    processed_shapes = [x.stem for x in processed_shapes_path.iterdir()]
    processed_shape_ids = [str(int(x.split('_')[0].split('shape')[1])) for x in processed_shapes]
    processed_shape_sids = [shape_meta[x]['source_id'] for x in processed_shape_ids]
    existing = []
    for x, y in tqdm(zip(processed_shapes, processed_shape_sids)):
        if (annotation_path / y).exists():
            # print(annotation_path / y, 'does not exist')
            existing.append(x)
    print(len(existing), '/', len(processed_shape_sids), 'exist')
    original_processed = processed_shapes_path
    pruned_processed = processed_shapes_pruned_path
    pruned_processed.mkdir(exist_ok=True)
    for x in tqdm(existing):
        shutil.copyfile(original_processed / f'{x}.pt', pruned_processed / f'{x}.pt')


def visualize_random_annotation():
    import random
    hex_to_rgb = lambda x: [int(x[i:i + 2], 16) for i in (1, 3, 5)]
    distinct_colors = ['#ff0000',  '#ffff00',  '#c71585',  '#00fa9a',  '#0000ff',  '#1e90ff',  '#ffdab9']
    # red, yellow, mediumvioletred, mediumspringgreen, blue, dodgerblue, peachpuff
    # chair/chair_head, chair/chair_back, chair/chair_arm, chair/chair_base, chair/chair_seat, chair/footrest, chair/other
    shape = random.choice(list(processed_shapes_pruned_path.iterdir())).stem
    npz = np.load(aligned_annotation_path / f'{shape}.npz')
    pts, labels = npz['pts'], npz['labels']
    labels = np.array([hex_to_rgb(distinct_colors[label]) for label in labels.tolist()])
    shutil.copyfile(mesh_dir / shape / "model_normalized.obj", "/home/nihalsid/Downloads/mesh.obj")
    mesh = trimesh.load(mesh_dir / shape / "model_normalized.obj", process=False)
    semantics = torch.load(processed_shapes_pruned_path / f'{shape}.pt')['semantics'].long()
    colored_semantics = torch.from_numpy(np.array([hex_to_rgb(distinct_colors[label]) for label in semantics.numpy().tolist()])).float()
    vertex_colors = torch.zeros(mesh.vertices.shape)
    torch_scatter.scatter_mean(colored_semantics.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3), torch.from_numpy(mesh.faces).reshape(-1).long(), dim=0, out=vertex_colors)
    trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=vertex_colors.numpy()).export("/home/nihalsid/Downloads/combined.obj")
    Path("/home/nihalsid/Downloads/annot.obj").write_text("\n".join([f"v {pts[i][0]} {pts[i][1]} {pts[i][2]} {labels[i][0]}  {labels[i][1]}  {labels[i][2]}" for i in range(pts.shape[0])]))


def normalize_rotate_points(annotation_dir, output_file):
    pts_lines = (annotation_dir / "pts-10000.txt").read_text().splitlines()
    labels_lines = (annotation_dir / "label-10000.txt").read_text().splitlines()
    id_map = process_result_json(annotation_dir / "result.json")
    pts = np.array([[float(y) for y in x.strip().split(' ')] for x in pts_lines]).astype(np.float32)
    labels = np.array([int(x.strip()) for x in labels_lines]).astype(np.int)
    labels = id_map[labels]
    assert (labels == -1).sum() == 0, annotation_dir
    rot_matrix = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    pts = (rot_matrix @ pts.T).T
    pts = pts - (pts.max(axis=0) + pts.min(axis=0)) / 2
    pts = pts / (pts.max(axis=0) - pts.min(axis=0)).max()
    np.savez(output_file, pts=pts, labels=labels)


def align_points_to_meshes():
    shape_meta = json.loads(shape_meta_path.read_text())
    pruned_processed = list(processed_shapes_pruned_path.iterdir())
    aligned_annotation_path.mkdir(exist_ok=True)
    for p in tqdm(pruned_processed):
        src_id = shape_meta[str(int(p.stem.split('_')[0].split('shape')[1]))]['source_id']
        normalize_rotate_points(annotation_path / src_id, aligned_annotation_path / f'{p.stem}.npz')


def unique_labels():
    unique_annots = set()
    for x in tqdm(aligned_annotation_path.iterdir()):
        labels = np.load(str(x))['labels']
        if labels.max() > 100:
            print(x)
        unique_annots.update(labels.tolist())
    print(sorted(list(unique_annots)))
    print(len(unique_annots), min(unique_annots), max(unique_annots))


def process_result_json(json_path):
    def process_hierarchy(current_node, current_dict, current_name):
        current_dict[current_node['id']] = f'{current_name}/{current_node["name"]}'
        if 'children' in current_node:
            for child in current_node['children']:
                process_hierarchy(child, current_dict, f'{current_name}/{current_node["name"]}')
    level_0_ids = [f'/{x}' for x in level_0_path.read_text().splitlines()]
    hierarchy = json.loads(json_path.read_text())
    result = {}
    for node in hierarchy:
        process_hierarchy(node, result, "")
    id_map = np.ones(500, dtype=np.int32) * -1
    for k in list(result.keys()):
        if result[k] == '/chair':
            continue
        prefix = '/'.join(result[k].split('/')[:3])
        if prefix in level_0_ids:
            id_map[k] = level_0_ids.index(prefix)
        else:
            print(prefix)
    return id_map


def map_mesh_faces_to_labels(proc_files):
    from sklearn.neighbors import KDTree
    from scipy.stats import mode
    for p in tqdm(proc_files):
        mesh = trimesh.load(mesh_dir / p.stem / "model_normalized.obj", process=False)
        face_centers = mesh.triangles.mean(axis=1)
        npz = np.load(aligned_annotation_path / f'{p.stem}.npz')
        pts, labels = npz['pts'], npz['labels']
        kdt = KDTree(pts, metric='euclidean')
        indices = kdt.query(face_centers, k=5, return_distance=False)
        most_frequent_label = mode(labels[indices], axis=1)[0].flatten()
        pt_arxiv = torch.load(p)
        pt_arxiv['semantics'] = torch.from_numpy(most_frequent_label).float()
        torch.save(pt_arxiv, p)


def add_semantics():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)

    args = parser.parse_args()
    files = sorted([x for x in processed_shapes_pruned_path.iterdir()])
    files = [x for i, x in enumerate(files) if i % args.num_proc == args.proc]

    map_mesh_faces_to_labels(files)


if __name__ == "__main__":
    # process_result_json(Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/part-net/annotations/85f4e9037425401a191c3762b497eca9/result_after_merging.json"))
    # normalize_rotate_points(Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape-model/part-net/annotations/85f4e9037425401a191c3762b497eca9"), "test")
    # align_points_to_meshes()
    # add_semantics()
    visualize_random_annotation()

