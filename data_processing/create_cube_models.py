import shutil
from argparse import ArgumentParser
from shutil import copyfile
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import random
import json

from dataset.texture_map_dataset import TextureMapDataset
from util.feature_loss import FeatureLossHelper
from util.misc import read_list, write_list, normalize_npy


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


def find_closest_texture_in_train(dataset):
    import torch
    to_torch_tensor = lambda x: torch.from_numpy(x).permute((2, 0, 1)).unsqueeze(0)
    dataset_0, dataset_1 = dataset.split('/')
    target_folder = f"data/{dataset_0}-model/{dataset_1}"
    split_folder = f"data/splits/{dataset_0}/{dataset_1}"
    train = read_list(f"{split_folder}/official/train.txt")
    val = read_list(f"{split_folder}/official/val_vis.txt")
    im_k_dict = {}

    from_rgb = TextureMapDataset.convert_cspace("lab")[0]
    mse_loss = torch.nn.MSELoss(reduction='mean')

    feature_loss_helper = FeatureLossHelper(['relu4_2'], ['relu3_2', 'relu4_2'])
    feature_loss_helper.move_to_device(torch.device('cuda:0'))

    total_error = 0.

    for tex_k in tqdm(train):
        im_k = normalize_npy(from_rgb(np.array(Image.open(Path(target_folder) / tex_k / "texture.png")).astype(np.float32)), "lab")
        im_k_dict[tex_k] = im_k

    best_match_dict = {}
    for tex_q in tqdm(val):
        im_q = normalize_npy(from_rgb(np.array(Image.open(Path(target_folder) / tex_q / "texture.png")).astype(np.float32)), "lab")
        min_error = 256*256*256
        best_match = ""
        for tex_k in train:
            im_k = im_k_dict[tex_k]
            error = mse_loss(to_torch_tensor(im_q)[:, 0:1, :, :], to_torch_tensor(im_k)[:, 0:1, :, :]).item()
            # error = feature_loss_helper.calculate_feature_loss(to_torch_tensor(im_q).float().cuda()[:, 0:1, :, :], to_torch_tensor(im_k).float().cuda()[:, 0:1, :, :]).mean().item()
            # style_loss_maps = feature_loss_helper.calculate_style_loss(to_torch_tensor(im_q).float().cuda()[:, 0:1, :, :], to_torch_tensor(im_k).float().cuda()[:, 0:1, :, :])
            # error = (style_loss_maps[0].mean() + style_loss_maps[1].mean()).item()
            if error < min_error:
                min_error = error
                best_match = tex_k
        total_error += min_error
        best_match_dict[tex_q] = best_match
    Path(split_folder, "closest_train.json").write_text(json.dumps(best_match_dict))
    print('Average opt. error:', total_error / len(val))


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


def create_split_cubetextures():
    base_folder = "data/SingleShape-model/CubeTextures/base"
    for split in ['train', 'val']:
        items = list('.'.join(x.name.split('.')[:-1]) for x in Path(base_folder, 'textures', split).iterdir())
        write_list(f'data/splits/SingleShape/CubeTextures/official/{split}.txt', items)
    val = read_list('data/splits/SingleShape/CubeTextures/official/val.txt')
    write_list('data/splits/SingleShape/CubeTextures/official/val_vis.txt', random.sample(val, 50))
    train = read_list('data/splits/SingleShape/CubeTextures/official/train.txt')
    write_list('data/splits/SingleShape/CubeTextures/official/train_vis.txt', random.sample(train, 25))
    write_list('data/splits/SingleShape/CubeTextures/official/train_val.txt', random.sample(train, int(len(train) * 0.20)))


def create_partial_textures(dataset, num_views, proc, num_proc):
    from scipy.spatial import cKDTree
    dataset_0, dataset_1 = dataset.split('/')
    folder = Path(f"data/{dataset_0}/{dataset_1}")
    all_list = read_list(f'data/splits/{dataset_0}/{dataset_1}/official/all.txt')
    all_list = [x for i, x in enumerate(all_list) if i % num_proc == proc]

    with Image.open(folder / all_list[0] / f"noc.png") as noc_img:
        noc = np.array(noc_img)[:, :, :3]
    mask_texture = np.logical_not(np.logical_and(np.logical_and(noc[:, :, 0] == 0, noc[:, :, 1] == 0), noc[:, :, 2] == 0))
    noc_flat = noc.reshape((-1, 3))
    mask_texture_flat = mask_texture.squeeze().ravel()
    indices_noc_flat = np.array(list(range(noc_flat.shape[0])))
    noc_flat = noc_flat[mask_texture_flat, :]
    indices_noc_flat = indices_noc_flat[mask_texture_flat]
    kd_tree = cKDTree(noc_flat)

    for sample in tqdm(all_list):
        if (folder / sample / "surface_texture.png").exists():
            for i in range(num_views):
                with Image.open(folder / sample / f"rgb_{i:03d}.png") as render_img:
                    render = np.array(render_img)[:, :, :3]
                with Image.open(folder / sample / f"noc_render_{i:03d}.png") as noc_render_img:
                    noc_render = np.array(noc_render_img)[:, :, :3]
                with Image.open(folder / sample / f"silhoutte_{i:03d}.png") as mask_render_img:
                    mask_render = np.array(mask_render_img)
                    mask_render = np.logical_and(np.logical_and(mask_render[:, :, 0] < 50, mask_render[:, :, 1] < 50), mask_render[:, :, 2] < 50)
                render_flat = render.reshape((-1, 3))
                noc_render_flat = noc_render.reshape((-1, 3))
                mask_render_flat = mask_render.squeeze().ravel()
                render_flat = render_flat[mask_render_flat, :]
                noc_render_flat = noc_render_flat[mask_render_flat, :]
                _dist, indices = kd_tree.query(noc_render_flat, k=1)
                closest_indices = indices_noc_flat[indices]
                partial_texture = np.zeros_like(noc).reshape((-1, 3))
                partial_texture[closest_indices, :] = render_flat
                Image.fromarray(partial_texture.reshape(noc.shape)).save(folder / sample / f"partial_texture_{i:03d}.png")


def create_partial_textures_reverse_lookup(dataset, num_views, proc, num_proc):
    from scipy.spatial import cKDTree
    dataset_0, dataset_1 = dataset.split('/')
    folder = Path(f"data/{dataset_0}/{dataset_1}")
    all_list = read_list(f'data/splits/{dataset_0}/{dataset_1}/official/all.txt')
    all_list = [x for i, x in enumerate(all_list) if i % num_proc == proc]

    for sample in tqdm(all_list):
        if (folder / sample / "surface_texture.png").exists():
            print(sample)
            for i in range(num_views):
                with Image.open(folder / sample / f"rgb_{i:03d}.png") as render_img:
                    render = np.array(render_img)[:, :, :3]
                with Image.open(folder / sample / f"noc_render_{i:03d}.png") as noc_render_img:
                    noc_render = np.array(noc_render_img)[:, :, :3]
                with Image.open(folder / sample / f"silhoutte_{i:03d}.png") as mask_render_img:
                    mask_render = np.array(mask_render_img)
                    mask_render = np.logical_and(np.logical_and(mask_render[:, :, 0] < 50, mask_render[:, :, 1] < 50), mask_render[:, :, 2] < 50)
                render_flat = render.reshape((-1, 3))
                noc_render_flat = noc_render.reshape((-1, 3))
                indices_noc_render_flat = np.array(list(range(noc_render_flat.shape[0])))
                mask_render_flat = mask_render.squeeze().ravel()
                noc_render_flat = noc_render_flat[mask_render_flat, :]
                kd_tree = cKDTree(noc_render_flat)
                indices_noc_render_flat = indices_noc_render_flat[mask_render_flat]

                with Image.open(folder / sample / f"noc.png") as noc_img:
                    noc = np.array(noc_img)[:, :, :3]
                mask_texture = np.logical_not(np.logical_and(np.logical_and(noc[:, :, 0] == 0, noc[:, :, 1] == 0), noc[:, :, 2] == 0))
                noc_flat = noc.reshape((-1, 3))
                indices_noc_flat = np.array(list(range(noc_flat.shape[0])))
                mask_texture_flat = mask_texture.squeeze().ravel()
                noc_flat = noc_flat[mask_texture_flat, :]
                indices_noc_flat = indices_noc_flat[mask_texture_flat]

                dist, indices = kd_tree.query(noc_flat, k=1)
                closest_indices = indices_noc_render_flat[indices]
                partial_texture = np.zeros_like(noc).reshape((-1, 3))
                partial_texture_mask = np.zeros_like(noc).reshape((-1, 3))
                partial_texture[indices_noc_flat, :] = render_flat[closest_indices, :]
                partial_texture[indices_noc_flat[dist > 5], :] = 0
                partial_texture_mask[indices_noc_flat, :] = 1
                partial_texture_mask[indices_noc_flat[dist > 5], :] = 0
                Image.fromarray(partial_texture.reshape(noc.shape)).save(folder / sample / f"inv_partial_texture_{i:03d}.png")
                Image.fromarray((partial_texture_mask * 255).astype(np.uint8).reshape(noc.shape)).save(folder / sample / f"inv_partial_mask_{i:03d}.png")


def create_texture_completion_task_data(dataset, patch_range, proc, num_proc):
    import torch
    from dataset.texture_map_dataset import TextureMapDataset
    dataset_0, dataset_1 = dataset.split('/')
    folder = Path(f"data/{dataset_0}/{dataset_1}")
    output_folder = Path(f"data/{dataset_0}/completion_{dataset_1}")
    all_list = read_list(f'data/splits/{dataset_0}/{dataset_1}/official/all.txt')
    all_list = [x for i, x in enumerate(all_list) if i % num_proc == proc]
    with Image.open(folder / all_list[0] / "surface_normals.png") as normal_img:
        arr = np.array(normal_img)
        mask = torch.from_numpy(np.logical_or(np.logical_or(arr[:, :, 0] != 0, arr[:, :, 1] != 0), arr[:, :, 2] != 0)).unsqueeze(0).unsqueeze(0).float()
    mask_arr = (mask.squeeze().numpy() * 255).astype(np.uint8)
    sampling_area = np.nonzero(TextureMapDataset.get_valid_sampling_area(mask, np.max(patch_range)).squeeze().numpy())

    for sample in tqdm(all_list):
        (output_folder / sample).mkdir(exist_ok=True, parents=True)
        with Image.open(folder / sample / f"surface_texture.png") as tex_img:
            texture = np.array(tex_img)[:, :, :3]
        shutil.copyfile(folder / sample / f"surface_texture.png", output_folder / sample / "texture_complete.png")
        Image.fromarray(mask_arr).save(output_folder / sample / "mask.png")
        for frame_idx in range(12):
            missing_mask_arr = np.zeros_like(mask_arr)
            missing_texture_arr = np.copy(texture)
            patch_idx = random.choice(range(sampling_area[0].shape[0]))
            patch_size = random.choice(range(patch_range[0], patch_range[1]))
            missing_texture_arr[sampling_area[0][patch_idx]: sampling_area[0][patch_idx] + patch_size, sampling_area[1][patch_idx]: sampling_area[1][patch_idx] + patch_size, :] = 0
            missing_mask_arr[sampling_area[0][patch_idx]: sampling_area[0][patch_idx] + patch_size, sampling_area[1][patch_idx]: sampling_area[1][patch_idx] + patch_size] = 255
            Image.fromarray(missing_texture_arr).save(output_folder / sample / f"texture_incomplete_{frame_idx:02d}.png")
            Image.fromarray(missing_mask_arr).save(output_folder / sample / f"missing_{frame_idx:02d}.png")


def run_partial_texture_proc():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='SingleShape/CubeDice', type=str)
    parser.add_argument('--num_views', default=12, type=int)
    parser.add_argument('--num_proc', default=1, type=int)
    parser.add_argument('--proc', default=0, type=int)
    args = parser.parse_args()
    create_partial_textures_reverse_lookup(args.dataset, args.num_views, args.proc, args.num_proc)


def run_texture_completion_proc():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='SingleShape/CubeTextures', type=str)
    parser.add_argument('--patch_range', nargs='+', default=[70, 85])
    parser.add_argument('--num_proc', default=1, type=int)
    parser.add_argument('--proc', default=0, type=int)
    args = parser.parse_args()
    create_texture_completion_task_data(args.dataset, args.patch_range, args.proc, args.num_proc)


def run_find_closest_texture():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='SingleShape/CubeTextures', type=str)
    args = parser.parse_args()
    find_closest_texture_in_train(args.dataset)


if __name__ == "__main__":
    run_find_closest_texture()
