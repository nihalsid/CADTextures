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
        failure_lr = subprocess.call(f'{"./quadriflow"} {args[0][1]}', shell=True)
        if failure_lr != 0:
            print("Retrying...")
            new_args = " ".join(args[0][1].split(" ")[1:])
            failure_lr = subprocess.call(f'{"./quadriflow"} {new_args}', shell=True)
            if failure_lr != 0:
                raise Exception(f"error code {failure_lr} for {new_args}")
    except Exception as err:
        print("ERROR:", err)
        Path(args[0][0] + '.txt').write_text(str(err))
    os.chdir('/tmp')
    shutil.rmtree(tmpdir)


def quadriflow_wrapper(input_path):
    face_to_res = {
        96: [16, 8],
        384: [32, 24, 16, 8],
        1536: [64, 56, 48, 40, 32, 24, 16],
        6144: [128, 96, 88, 80, 72, 64, 56, 48, 40, 32]
    }
    collected_args = []
    for fcount, resolutions in face_to_res.items():
        for res in resolutions:
            output_filename = f'quad_{fcount:05d}_{res:03d}'
            quad_dirname = f"{input_path.name}_{output_filename}"
            # if it hangs, try without -sat and -mcf or with only one of them
            args = ["-sat", '-mcf', "-i", f"{str(input_path / f'{res:03d}.obj')}", '-o', f"{str(input_path / output_filename)}.obj", "-f", f"{fcount}"]
            collected_args.append([quad_dirname, ' '.join(args)])

    with Pool(8) as p:
        p.map(quadriflow_executor, collected_args)


def check_existence(root_files):
    face_to_res = {
        96: [16, 8],
        384: [32, 24, 16, 8],
        1536: [64, 56, 48, 40, 32, 24, 16],
        6144: [128, 96, 88, 80, 72, 64, 56, 48, 40, 32]
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
        # if it hangs, try without -sat and -mcf or with only one of them
        args = ["-sat", '-mcf', "-i", f"{str(path.parent / f'{R:03d}.obj')}", '-o', f"{str(path.parent / output_filename)}.obj", "-f", f"{target_faces}"]
        quadriflow_executor([quad_dirname, ' '.join(args)])
        num_faces = trimesh.load(path, process=False).faces.shape[0]


def get_num_faces(root_files):
    greater, equal, less = [], [], []
    for input_path in tqdm(root_files):
        qmeshes = [x for x in input_path.iterdir() if x.name.startswith("quad_")]
        for qmesh in qmeshes:
            mesh = trimesh.load(qmesh, process=False)
            supposed_num_faces = int(qmesh.name.split('_')[1])
            actual_num_faces = mesh.faces.shape[0]
            if actual_num_faces == supposed_num_faces:
                equal.append(qmesh)
            elif actual_num_faces > supposed_num_faces:
                greater.append(qmesh)
            else:
                less.append(qmesh)
    print("get_num_faces:", len(greater), len(equal), len(less), len(greater) + len(equal) + len(less))
    return greater, less


def check_for_nans(root_files):
    import numpy as np
    nan_meshes = []
    for input_path in tqdm(root_files):
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

    non_existent_files = check_existence(files)

    for f in tqdm(non_existent_files):
        quadriflow_wrapper(f)

    nan_meshes = check_for_nans(files)
    for f in nan_meshes:
        os.remove(str(f))

    greater, less = get_num_faces(files)

    for f in tqdm(greater):
        retopologize_file(f)

