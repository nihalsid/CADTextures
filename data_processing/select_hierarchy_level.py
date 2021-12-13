from pathlib import Path
import trimesh
import torch
import json


def select_hierarchy_level(path):
    from chamferdist import ChamferDistance
    face_to_res = {
        96: [[16, 8], 24],
        384: [[32, 24, 16], 40],
        1536: [[64, 56, 48, 40, 32], 72],
        6144: [[96, 88, 80, 72, 64], 128]
    }
    if (path / "selection.json").exists():
        return
    num_samples = 50000
    chamfer = ChamferDistance()
    choosen_face_res = {}
    for face_num in face_to_res.keys():
        choosen_res = -1
        best_cd = 1e5

        for res_to_check in face_to_res[face_num][0]:
            try:
                # reference_mesh = trimesh.load(path / f"{face_to_res[face_num][1]:03d}.obj")
                reference_mesh = trimesh.load(path / f"{res_to_check:03d}.obj", process=False)
                samples_ref = torch.from_numpy(trimesh.sample.sample_surface_even(reference_mesh, num_samples)[0]).unsqueeze(0).cuda().float()

                current_mesh = trimesh.load(path / f"quad_{face_num:05d}_{res_to_check:03d}.obj", process=False)
                samples_cur = torch.from_numpy(trimesh.sample.sample_surface_even(current_mesh, num_samples)[0]).unsqueeze(0).cuda().float()
                cd = chamfer(samples_cur, samples_ref).item()
                if cd < best_cd:
                    choosen_res = res_to_check
                    best_cd = cd
            except Exception as err:
                print(path, face_num, res_to_check)
                print(err)
        choosen_face_res[face_num] = choosen_res
        # print(f'Choosen mesh for {face_num}: {choosen_res} with cd {best_cd}')

    (path / "selection.json").write_text(json.dumps(choosen_face_res))


def make_cube_24(path):
    import shutil
    shutil.copyfile(Path("/rhome/ysiddiqui/CADTextures/data_processing/blender/quad_00024_001.obj"), path / "quad_00024_001.obj")


def copy_to_meshdir(path, destination):
    import shutil
    selection = json.loads((path / "selection.json").read_text())["6144"]
    (destination / path.name).mkdir(exist_ok=True, parents=True)
    shutil.copyfile(path / f"quad_06144_{selection:03d}.obj", destination / path.name / "model_normalized.obj")


def pad_faces(path):
    N, R = [int(x) for x in path.name.split('.')[0].split('_')[1:3]]
    mesh = trimesh.load(path, process=False)
    num_vertices = mesh.vertices.shape[0]
    num_faces = mesh.faces.shape[0]
    if N - num_faces > 0:
        new_vertices = '\n'.join(["v 0 0 0", "v 0.0001 0 0", "v 0 0.0001 0", "v 0.0001 0.0001 0"])
        new_faces = '\n'.join([f"f {num_vertices + 1} {num_vertices + 2} {num_vertices + 3} {num_vertices + 4}"] * (N - num_faces))
        path.write_text(path.read_text() + "\n" + new_vertices + "\n" + new_faces)


def get_less_faces(root_files):
    less = []
    for input_path in tqdm(root_files):
        selections = json.loads((input_path / "selection.json").read_text())
        qmeshes = [f"quad_{int(k):05d}_{v:03d}.obj" for k, v in selections.items()]
        for qmesh in [input_path / x for x in qmeshes]:
            mesh = trimesh.load(qmesh, process=False)
            supposed_num_faces = int(qmesh.name.split('_')[1])
            actual_num_faces = mesh.faces.shape[0]
            if actual_num_faces < supposed_num_faces:
                less.append(qmesh)
    return less


def unexpected_bound_meshes():
    import math

    unexpected = []
    for f in tqdm(files):
        selections = json.loads((f / "selection.json").read_text())
        selection_keys = sorted(list(selections.keys()), key=lambda x: int(x), reverse=True)
        qmeshes = [f"quad_{int(k):05d}_{selections[k]:03d}.obj" for k in selection_keys]
        for idx, qmesh in enumerate(qmeshes):
            scale_tol = 0.1 + 0.425 * ((idx + 1) / len(qmeshes))
            mesh = trimesh.load(f / qmesh, process=False)
            center = (mesh.bounds[0] + mesh.bounds[1]) / 2
            scale = (mesh.bounds[1] - mesh.bounds[0]).max()
            for v0, v1, tol in zip([center[0], center[1], center[2], scale], [0., 0., 0., 1], [0.125, 0.125, 0.125, scale_tol]):
                if not math.isclose(v0, v1, abs_tol=tol):
                    print(f / qmesh, v0, v1, tol)
                    unexpected.append(f / qmesh)
                    break

    print("Unexpected: ", len(unexpected))


if __name__ == '__main__':
    from tqdm import tqdm
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', type=str)
    parser.add_argument('-n', '--num_proc', default=1, type=int)
    parser.add_argument('-p', '--proc', default=0, type=int)

    args = parser.parse_args()
    input_folder = Path(args.input_folder)
    files = sorted([x for x in input_folder.iterdir()])
    files = [x for i, x in enumerate(files) if i % args.num_proc == args.proc]

    destination_folder = input_folder.parents[1] / input_folder.parent.name.split("-model")[0] / input_folder.name
    print(destination_folder)

    for f in tqdm(files):
        select_hierarchy_level(f)
        make_cube_24(f)

    less = get_less_faces(files)
    print("LessFaces: ", len(less))

    for f in tqdm(less):
        pad_faces(f)

    for f in tqdm(files):
        copy_to_meshdir(f, destination_folder)

    unexpected_bound_meshes()
