from argparse import ArgumentParser
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from util.misc import read_list


def sample_textures(dataset, proc, num_proc):
    debug_vis = False

    dataset_0, dataset_1 = dataset.split('/')
    folder = Path(f"data/{dataset_0}/{dataset_1}")
    all_list = read_list(f'data/splits/{dataset_0}/{dataset_1}/official/all.txt')
    all_list = [x for i, x in enumerate(all_list) if i % num_proc == proc]
    sigma = [0., 1., 2., 3., 5., 8., 13., 21., 34.]

    with Image.open(folder / all_list[0] / "surface_normals.png") as normal_img:
        arr = np.array(normal_img)
        mask = torch.from_numpy(np.logical_or(np.logical_or(arr[:, :, 0] != 0, arr[:, :, 1] != 0), arr[:, :, 2] != 0)).unsqueeze(0).unsqueeze(0).float()
    mask_arr = (mask.squeeze().numpy() * 255).astype(np.uint8)
    valid_pix = np.nonzero(mask_arr)

    for sample in tqdm(all_list):
        with Image.open(folder / sample / f"surface_texture.png") as tex_img:
            texture = np.array(tex_img)[:, :, :3]
        points = []
        samples = []
        for s in sigma:
            sigma_points = (-0.5 + (valid_pix[0] + np.random.normal(0, 1, valid_pix[0].shape[0]) * s) / mask_arr.shape[0], -0.5 + (valid_pix[1] + np.random.normal(0, 1, valid_pix[1].shape[0]) * s) / mask_arr.shape[1])
            mask = np.logical_and(np.logical_and(sigma_points[0] > -0.5, sigma_points[0] < 0.5), np.logical_and(sigma_points[1] > -0.5, sigma_points[1] < 0.5))
            sigma_points = (sigma_points[0][mask], sigma_points[1][mask])
            coordinates_r = np.clip(np.round((sigma_points[0] + 0.5) * mask_arr.shape[0]).astype(np.int32), 0, mask_arr.shape[0] - 1)
            coordinates_c = np.clip(np.round((sigma_points[1] + 0.5) * mask_arr.shape[1]).astype(np.int32), 0, mask_arr.shape[1] - 1)
            colors = texture[coordinates_r, coordinates_c, :]
            points.append(np.hstack([sigma_points[0][:, np.newaxis], sigma_points[1][:, np.newaxis]]))
            samples.append(colors)
        points = np.vstack(points)
        samples = np.vstack(samples)
        np.savez_compressed(folder / sample / f"implicit_samples.npz", points=points, values=samples)
        if debug_vis:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
            coordinates_r = np.clip(np.round((points[:, 0] + 0.5) * image.shape[0]).astype(np.int32), 0, image.shape[0] - 1)
            coordinates_c = np.clip(np.round((points[:, 1] + 0.5) * image.shape[1]).astype(np.int32), 0, image.shape[1] - 1)
            image[coordinates_r, coordinates_c, :] = samples[:, :]
            Image.fromarray(image).save("test.jpg")


def run_sampling_proc():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='SingleShape/CubeTextures', type=str)
    parser.add_argument('--num_proc', default=1, type=int)
    parser.add_argument('--proc', default=0, type=int)
    args = parser.parse_args()
    sample_textures(args.dataset, args.proc, args.num_proc)


if __name__ == '__main__':
    run_sampling_proc()
