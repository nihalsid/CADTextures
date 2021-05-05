import torch
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from dataset.texture_map_dataset import TextureMapDataset


def test_mask_generation():
    patch_size = 25 * 3
    with Image.open("data/SingleShape/Cube/000-000-143/surface_normals.png") as texture_im:
        arr = np.array(texture_im)
        mask = torch.from_numpy(np.logical_or(np.logical_or(arr[:, :, 0] != 0, arr[:, :, 1] != 0), arr[:, :, 2] != 0)).unsqueeze(0).unsqueeze(0).float()
    sampling_area = TextureMapDataset.get_valid_sampling_area(mask, patch_size)
    f, axarr = plt.subplots(1, 2, figsize=(8, 4))
    axarr[0].imshow(mask[0, 0, :, :].numpy())
    axarr[0].axis('off')
    axarr[1].imshow(sampling_area[0, 0, :, :].numpy())
    axarr[1].axis('off')
    plt.tight_layout()
    plt.show()


def test_sampled_patches():
    patch_size = 25 * 3
    num_patches = 4
    with Image.open("data/SingleShape/CubeTextures/dtd-cracked_cracked_0056/surface_normals.png") as texture_im:
        arr = np.array(texture_im)
        mask = torch.from_numpy(np.logical_or(np.logical_or(arr[:, :, 0] != 0, arr[:, :, 1] != 0), arr[:, :, 2] != 0)).unsqueeze(0).unsqueeze(0).float()
    with Image.open("data/SingleShape/CubeTextures/dtd-cracked_cracked_0056/surface_texture.png") as texture_im:
        arr = np.array(texture_im)[:, :, :3]
        texture_a = torch.from_numpy(arr).unsqueeze(0).permute((0, 3, 1, 2)).float()
    with Image.open("data/SingleShape/CubeTextures/dtd-cracked_cracked_0145/surface_texture.png") as texture_im:
        arr = np.array(texture_im)[:, :, :3]
        texture_b = torch.from_numpy(arr).unsqueeze(0).permute((0, 3, 1, 2)).float()
    sampled_a, sampled_b_0, sampled_b_1 = TextureMapDataset.sample_patches(mask, patch_size, num_patches, texture_a, texture_b, texture_b)
    print(sampled_a.shape, sampled_b_0.shape, sampled_b_1.shape)

    f, axarr = plt.subplots(num_patches, 3, figsize=(12, 4 * num_patches))
    for i in range(num_patches):
        axarr[i, 0].imshow(sampled_a[i, :, :, :].permute((1, 2, 0)).numpy() / 255)
        axarr[i, 0].axis('off')
        axarr[i, 1].imshow(sampled_b_0[i, :, :, :].permute((1, 2, 0)).numpy() / 255)
        axarr[i, 1].axis('off')
        axarr[i, 2].imshow(sampled_b_1[i, :, :, :].permute((1, 2, 0)).numpy() / 255)
        axarr[i, 2].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_mask_generation()