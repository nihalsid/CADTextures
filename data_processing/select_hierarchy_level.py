from pathlib import Path
import numpy as np
import trimesh
import torch
import json
# import external.chamfer_distance.chamfer3D.dist_chamfer_3D as dist_chamfer_3D


def select_hierarchy_level(path):
    from chamferdist import ChamferDistance
    face_to_res = {
        96: [[16, 12], 24],
        384: [[32, 24, 16], 40],
        1536: [[64, 48, 40, 32], 80],
        6144: [[96, 80, 64], 128]
    }
    num_samples = 50000
    # chamfer = dist_chamfer_3D.chamfer_3DDist()
    chamfer = ChamferDistance()
    choosen_face_res = {}
    for face_num in face_to_res.keys():
        choosen_res = -1
        best_cd = 1e5

        for res_to_check in face_to_res[face_num][0]:
            try:
                # reference_mesh = trimesh.load(path / f"{face_to_res[face_num][1]:03d}.obj")
                reference_mesh = trimesh.load(path / f"{res_to_check:03d}.obj")
                reference_mesh.apply_translation(np.array([-64, -64, -64]))
                reference_mesh.apply_scale(1 / 128)
                samples_ref = torch.from_numpy(trimesh.sample.sample_surface_even(reference_mesh, num_samples)[0]).unsqueeze(0).cuda().float()

                current_mesh = trimesh.load(path / f"quad_{face_num:05d}_{res_to_check:03d}.obj")
                current_mesh.apply_translation(np.array([-64, -64, -64]))
                current_mesh.apply_scale(1 / 128)
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


def resize_meshes(path):
    mesh_paths = [x for x in path.iterdir() if x.name.endswith('.obj') and x.name != 'model_normalized.obj' and x.name != "quad_00024_001.obj"]
    base_mesh_path = path / "128.obj"
    for mp in mesh_paths:
        try:
            mesh = trimesh.load(mp, process=False)
            if (mesh.bounds[1] - mesh.bounds[0]).max() > 32:
                mesh.apply_translation(np.array([-64, -64, -64]))
                mesh.apply_scale(1 / 128)
                mesh.export(mp)
        except Exception as err:
            print(mp)
            print(err)
    try:
        base_bounds = trimesh.load(base_mesh_path, process=False).bounds
        base_bounds = base_bounds[1] - base_bounds[0]
        base_arg_max = np.argmax(base_bounds)
    except Exception as err:
        print(path)
        print(err)

    for mp in mesh_paths:
        try:
            mesh = trimesh.load(mp, process=False)
            current_max = (mesh.bounds[1] - mesh.bounds[0])[base_arg_max]
            mesh.apply_scale(base_bounds[base_arg_max] / current_max)
            mesh.apply_scale(1 / base_bounds[base_arg_max])
            mesh.export(mp)
        except Exception as err:
            print(mp)
            print(err)


def make_cube_24(path):
    import shutil
    shutil.copyfile(Path("blender/quad_00024_001.obj"), path / "quad_00024_001.obj")


def copy_to_meshdir(path, destination):
    import shutil
    selection = json.loads((path / "selection.json").read_text())["6144"]
    (destination / path.name).mkdir(exist_ok=True, parents=True)
    shutil.copyfile(path / f"quad_06144_{selection:03d}.obj", destination / path.name / "model_normalized.obj")


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
        # resize_meshes(f)
        # make_cube_24(f)
        # copy_to_meshdir(f, destination_folder)
