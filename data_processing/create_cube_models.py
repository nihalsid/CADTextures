from shutil import copyfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm

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
    # create_cube_models_from_base()
    create_split()
