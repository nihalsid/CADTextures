from shutil import copyfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import json

from util.misc import read_list, write_list


def create_cube_models_from_base():
    base_folder = "data/SingleShape-model/Cube/base"
    items = read_list(f"{base_folder}/256_train.txt") + read_list(f"{base_folder}/256_val.txt")
    colors = [[int(y) for y in x.split(',')] for x in items]
    for c in tqdm(colors):
        output_folder = Path(base_folder).parent / f"{c[0]:03d}-{c[1]:03d}-{c[2]:03d}"
        output_folder.mkdir(exist_ok=True)
        for o in ["material.mtl", "normalized_model.obj"]:
            copyfile(Path(base_folder) / o, output_folder / o)
        Image.new('RGB', (384, 384), (c[0], c[1], c[2])).save(output_folder / "texture.png")


def create_textured_cube_models_from_base():
    base_folder = "data/SingleShape-model/CubeTextures/base"
    items = list(Path(base_folder, 'textures', "train").iterdir()) + list(Path(base_folder, 'textures', "val").iterdir())
    all_samples = []
    for item in tqdm(items):
        all_samples.append(f"{'.'.join(item.name.split('.')[:-1])}")
        output_folder = Path(base_folder).parent / f"{'.'.join(item.name.split('.')[:-1])}"
        output_folder.mkdir(exist_ok=True)
        for o in ["material.mtl", "normalized_model.obj"]:
            copyfile(Path(base_folder) / o, output_folder / o)
        copyfile(item, output_folder / "texture.png")
    write_list('data/splits/SingleShape/CubeTextures/official/all.txt', all_samples)


def create_cube_models_single_texture_from_base():
    target_folder = "data/SingleShape-model/CubeSingleTexture02"
    base_folder = "data/SingleShape-model/Cube/base"
    mask_array = np.array(Image.open("data/SingleShape-model/Cube/base/mask_check.png").resize((384, 384), resample=Image.NEAREST))
    items = read_list(f"{base_folder}/256_train.txt") + read_list(f"{base_folder}/256_val.txt")
    colors = [[int(y) for y in x.split(',')] for x in items]
    for c in tqdm(colors):
        output_folder = Path(target_folder) / f"{c[0]:03d}-{c[1]:03d}-{c[2]:03d}"
        output_folder.mkdir(exist_ok=True)
        for o in ["material.mtl", "normalized_model.obj"]:
            copyfile(Path(base_folder) / o, output_folder / o)
        image_array = np.array(Image.new('RGB', (384, 384), (c[0], c[1], c[2])))
        image_array[mask_array, 0] = 255
        image_array[mask_array, 1] = 255
        image_array[mask_array, 2] = 255
        Image.fromarray(image_array).save(output_folder / "texture.png")


def find_closest_texture_in_train():
    base_folder = "data/SingleShape-model/Cube/base"
    target_folder = "data/SingleShape-model/Cube"
    split_folder = "data/splits/SingleShape/Cube"
    train = read_list(f"{base_folder}/256_train.txt")
    val = read_list(f"{base_folder}/256_val.txt")
    im_k_dict = {}

    for tex_k in train:
        c_k = [int(y) for y in tex_k.split(',')]
        im_k = np.array(Image.open(Path(target_folder) / f"{c_k[0]:03d}-{c_k[1]:03d}-{c_k[2]:03d}" / "texture.png")).astype(np.float32)
        im_k_dict[f"{c_k[0]:03d}-{c_k[1]:03d}-{c_k[2]:03d}"] = im_k

    best_match_dict = {}
    for tex_q in tqdm(val):
        c_q = [int(y) for y in tex_q.split(',')]
        im_q = np.array(Image.open(Path(target_folder) / f"{c_q[0]:03d}-{c_q[1]:03d}-{c_q[2]:03d}" / "texture.png")).astype(np.float32)
        min_error = 256*256*256
        best_match = ""
        for tex_k in train:
            c_k = [int(y) for y in tex_k.split(',')]
            im_k = im_k_dict[f"{c_k[0]:03d}-{c_k[1]:03d}-{c_k[2]:03d}"]
            error = np.linalg.norm(im_q - im_k, axis=2).mean()
            if error < min_error:
                min_error = error
                best_match = f"{c_k[0]:03d}-{c_k[1]:03d}-{c_k[2]:03d}"
        best_match_dict[f"{c_q[0]:03d}-{c_q[1]:03d}-{c_q[2]:03d}"] = best_match
    Path(split_folder, "closest_train.json").write_text(json.dumps(best_match_dict))


def create_split():
    base_folder = "data/SingleShape-model/Cube/base"
    for split in ['train', 'val']:
        items = read_list(f"{base_folder}/256_{split}.txt")
        colors = [[int(y) for y in x.split(',')] for x in items]
        names = [f"{c[0]:03d}-{c[1]:03d}-{c[2]:03d}" for c in colors]
        write_list(f'data/splits/SingleShape/Cube/official/{split}.txt', names)
    copyfile('data/splits/SingleShape/Cube/official/val.txt', 'data/splits/SingleShape/Cube/official/val_vis.txt')
    copyfile('data/splits/SingleShape/Cube/official/train.txt', 'data/splits/SingleShape/Cube/official/train_val.txt')
    write_list('data/splits/SingleShape/Cube/official/train_vis.txt', read_list('data/splits/SingleShape/Cube/official/train.txt')[:16])


if __name__ == "__main__":
    create_textured_cube_models_from_base()
