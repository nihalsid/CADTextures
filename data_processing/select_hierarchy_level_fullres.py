from pathlib import Path
from tqdm import tqdm
import json

from data_processing.select_hierarchy_level import get_less_faces, pad_faces, unexpected_bound_meshes


def select_hierarchy_level(path):
    choosen_face_res = json.loads((path / "selection.json").read_text())
    choosen_face_res['24576'] = 272
    (path / "selection.json").write_text(json.dumps(choosen_face_res))


def copy_to_meshdir(path, destination):
    import shutil
    selection = json.loads((path / "selection.json").read_text())["24576"]
    (destination / path.name).mkdir(exist_ok=True, parents=True)
    shutil.copyfile(path / f"quad_24576_{selection:03d}.obj", destination / path.name / "model_normalized.obj")


def messed_up_meshes(files):
    import trimesh
    messed_meshes = []
    for path in tqdm(files):
        selection = json.loads((path / "selection.json").read_text())["24576"]
        x = trimesh.load(path / f"quad_24576_{selection:03d}.obj", process=False)
        if trimesh.triangles.area(x.triangles).max() > 0.0025:
            print(path)
            messed_meshes.append(path)
    print("Messed up meshes: ", len(messed_meshes))
    return messed_meshes


if __name__ == '__main__':
    import argparse
    import shutil

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

    less = get_less_faces(files)
    print("LessFaces: ", len(less))

    for f in tqdm(less):
        pad_faces(f)

    for f in tqdm(files):
        copy_to_meshdir(f, destination_folder)

    unexpected_bound_meshes(files)
    messed_mesh = messed_up_meshes(files)
    # for f in messed_mesh:
    #     shutil.rmtree(f)
