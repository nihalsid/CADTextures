import json
import shutil
from pathlib import Path
import subprocess
from multiprocessing import Pool
import os

import trimesh
from tqdm import tqdm

in_root = Path("/cluster_HDD/gondor/ysiddiqui/shapenet-chairs")
quadriflow_path = "/rhome/ysiddiqui/quadriflow/quadriflow"


def copy_with_permissions(source, target):
    shutil.copy2(source, target)
    st = os.stat(source)
    os.chown(target, st.st_uid, st.st_gid)


def photoshape_exemplars():
    destination = Path("/cluster/gimli/ysiddiqui/CADTextures/Photoshape/exemplars")
    all_exemplars = [x for x in in_root.iterdir() if x.name.endswith('.exemplar.jpg')]
    destination.mkdir(exist_ok=True, parents=True)
    for x in all_exemplars:
        shutil.copyfile(x, destination / x.name.replace('.exemplar.jpg', '.jpg'))


def quadriflow_executor(*args):
    tmpdir = Path(f'/tmp/{args[0][0]}')
    tmpdir.mkdir(exist_ok=True)
    copy_with_permissions(quadriflow_path, tmpdir / "quadriflow")
    try:
        os.chdir(str(tmpdir))
        subprocess.call(f'{"./quadriflow"} {args[0][1]}', shell=True)
    except Exception as err:
        print("ERROR:", err)
        Path(args[0][0] + '.txt').write_text(str(err))
    os.chdir('/tmp')
    shutil.rmtree(tmpdir)


def quadriflow_wrapper(input_path):
    face_to_res = {
        96: [16, 12],
        384: [32, 24, 16, 12],
        1536: [64, 48, 40, 32, 24, 16],
        6144: [128, 96, 80, 64, 48, 40, 32]
    }
    # face_to_res = {
    #     384: [12],
    #     1536: [24, 16],
    #     6144: [48, 40, 32]
    # }
    collected_args = []
    for fcount, resolutions in face_to_res.items():
        for res in resolutions:
            output_filename = f'quad_{fcount:05d}_{res:03d}'
            quad_dirname = f"{input_path.name}_{output_filename}"
            args = ["-sat", '-mcf', "-i", f"{str(input_path / f'{res:03d}.obj')}", '-o', f"{str(input_path / output_filename)}.obj", "-f", f"{fcount}"]
            collected_args.append([quad_dirname, ' '.join(args)])

    with Pool(8) as p:
        p.map(quadriflow_executor, collected_args)


def check_existence(root_files):
    face_to_res = {
        96: [16, 12],
        384: [32, 24, 16, 12],
        1536: [64, 48, 40, 32, 24, 16],
        6144: [128, 96, 80, 64, 48, 40, 32]
    }
    empties = []
    for input_path in tqdm(root_files):
        for fcount, resolutions in face_to_res.items():
            for res in resolutions:
                output_filename = f'quad_{fcount:05d}_{res:03d}'
                if not (input_path / (output_filename + '.obj')).exists() or (input_path / (output_filename + '.obj')).read_text().strip() == '':
                    # print(input_path / (output_filename + '.obj'))
                    empties.append(input_path)
    empties = list(set(empties))
    return empties


def retopologize_file(path):
    N, R = [int(x) for x in path.name.split('.')[0].split('_')[1:3]]
    num_faces = trimesh.load(path, process=False).faces.shape[0]
    target_faces = N
    decrement = (num_faces - N) // 2 + 1
    while num_faces - N > 0:
        print('NumFaces: ', num_faces, 'TargetFaces: ', target_faces)
        target_faces = target_faces - decrement
        output_filename = path.name.split('.')[0]
        quad_dirname = f"{path.parent.name}_{output_filename}"
        args = ["-sat", '-mcf', "-i", f"{str(path.parent / f'{R:03d}.obj')}", '-o', f"{str(path.parent / output_filename)}.obj", "-f", f"{target_faces}"]
        quadriflow_executor([quad_dirname, ' '.join(args)])
        num_faces = trimesh.load(path, process=False).faces.shape[0]


def get_num_faces(root_files):
    greater, equal, less = [], [], []
    for input_path in tqdm(root_files):
        selections = json.loads((input_path / "selection.json").read_text())
        qmeshes = [f"quad_{int(k):05d}_{v:03d}.obj" for k, v in selections.items()]
        for qmesh in [input_path / x for x in qmeshes]:
            mesh = trimesh.load(qmesh, process=False)
            supposed_num_faces = int(qmesh.name.split('_')[1])
            actual_num_faces = mesh.faces.shape[0]
            if actual_num_faces == supposed_num_faces:
                equal.append(qmesh)
            elif actual_num_faces > supposed_num_faces:
                greater.append(qmesh)
            else:
                less.append(qmesh)
    print(len(greater), len(equal), len(less), len(greater) + len(equal) + len(less))
    # Path("greater.txt").write_text("\n".join(greater))
    # Path("equal.txt").write_text("\n".join(equal))
    # Path("less.txt").write_text("\n".join(less))
    return greater, less


def pad_faces(path):
    N, R = [int(x) for x in path.name.split('.')[0].split('_')[1:3]]
    mesh = trimesh.load(path, process=False)
    num_vertices = mesh.vertices.shape[0]
    num_faces = mesh.faces.shape[0]
    if N - num_faces > 0:
        new_vertices = '\n'.join(["v 0 0 0", "v 0.0001 0 0", "v 0 0.0001 0", "v 0.0001 0.0001 0"])
        new_faces = '\n'.join([f"f {num_vertices + 1} {num_vertices + 2} {num_vertices + 3} {num_vertices + 4}"] * (N - num_faces))
        path.write_text(path.read_text() + "\n" + new_vertices + "\n" + new_faces)


def check_for_nans(root_files):
    import numpy as np
    nan_meshes = []
    for input_path in tqdm(root_files):
        # selections = json.loads((input_path / "selection.json").read_text())
        # meshes = [input_path / f"quad_{int(k):05d}_{v:03d}.obj" for k, v in selections.items()]
        meshes = [x for x in input_path.iterdir() if x.name.startswith("quad_")]
        for m in meshes:
            mesh = trimesh.load(m, process=False)
            if np.isnan(np.sum(mesh.vertices)):
                nan_meshes.append(m)
    return nan_meshes


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str)
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)

    args = parser.parse_args()

    files = sorted([x for x in Path(args.input_folder).iterdir()])
    files = [x for i, x in enumerate(files) if i % args.num_proc == args.proc]

    # files = check_existence(files)

    # for f in tqdm(files):
    #     quadriflow_wrapper(f)

    # greater, less = get_num_faces(files)

    # nan_meshes = check_for_nans(files)
    # delete nan meshes

    # for f in tqdm(greater):
    #     retopologize_file(f)

    # for f in tqdm(less):
    #     pad_faces(f)
