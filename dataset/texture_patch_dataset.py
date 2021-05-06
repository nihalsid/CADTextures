import torch
from pathlib import Path
import numpy as np
import random
from PIL import Image
import cv2 as cv

from dataset.texture_map_dataset import TextureMapDataset
from util.misc import move_batch_to_gpu, apply_batch_color_transform_and_normalization, denormalize_and_rgb


class TexturePatchDataset(torch.utils.data.Dataset):

    def __init__(self, config, shape, view_index, patch_size, dataset_size):
        super().__init__()
        self.shape = shape
        self.view_index = view_index
        self.texture_map_size = config.dataset.texture_map_size
        self.color_space = config.dataset.color_space
        self.path_to_dataset = Path(config.dataset.data_dir) / config.dataset.name
        self.from_rgb, self.to_rgb = TextureMapDataset.convert_cspace(config.dataset.color_space)
        with Image.open(self.path_to_dataset / shape / 'surface_texture.png') as texture_im:
            self.target_texture = self.from_rgb(TextureMapDataset.process_to_padded_thumbnail(texture_im, self.texture_map_size)).astype(np.float32)
        with Image.open(self.path_to_dataset / shape / 'noc.png') as noc_im:
            noc = TextureMapDataset.process_to_padded_thumbnail(noc_im, self.texture_map_size)
            self.mask_texture = np.logical_not(np.logical_and(np.logical_and(noc[:, :, 0] == 0, noc[:, :, 1] == 0), noc[:, :, 2] == 0))
        with Image.open(self.path_to_dataset / self.shape / f"inv_partial_texture_{self.view_index:03d}.png") as texture_im:
            self.partial_texture = self.from_rgb(TextureMapDataset.process_to_padded_thumbnail(texture_im, self.texture_map_size)).astype(np.float32)
        with Image.open(self.path_to_dataset / self.shape / f"inv_partial_mask_{self.view_index:03d}.png") as mask_im:
            mask_im.thumbnail((self.texture_map_size, self.texture_map_size))
            self.partial_mask = np.array(mask_im)[:, :, 0] > 0
        self.patch_size = patch_size
        kernel_size = patch_size
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size), (kernel_size // 2, kernel_size // 2))
        self.mask_eroded_input = np.array(cv.erode(self.partial_mask.astype(np.uint8), kernel))
        self.non_zero_input = np.nonzero(self.mask_eroded_input)
        kernel_size = patch_size * 3 // 2
        kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size), (kernel_size // 2, kernel_size // 2))
        self.mask_eroded_target = np.array(cv.erode(self.mask_texture.astype(np.uint8), kernel))
        self.non_zero_target = np.nonzero(self.mask_eroded_target)
        self.mask_missing = np.logical_and(np.logical_not(self.partial_mask), self.mask_texture)
        self.size = dataset_size
        print('Input Patches:', self.non_zero_input[0].shape[0])
        print('Generated Patches:', self.non_zero_target[0].shape[0])

    @staticmethod
    def move_batch_to_gpu(batch, device):
        move_batch_to_gpu(batch, device, ['target', 'input', 'mask_texture', 'mask_missing', 'input_patch'])

    def apply_batch_transforms(self, batch):
        apply_batch_color_transform_and_normalization(batch, ['input', 'target', 'input_patch'], [], self.color_space)
        batch['target'] = TextureMapDataset.apply_mask_texture(batch['target'], batch['mask_texture'])

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        idx = random.randint(0, self.non_zero_input[0].shape[0] - 1)
        patch = self.partial_texture[self.non_zero_input[0][idx]: self.non_zero_input[0][idx] + self.patch_size, self.non_zero_input[1][idx]: self.non_zero_input[1][idx] + self.patch_size, :]
        return {
            'name': f'{index:05d}',
            'target': np.ascontiguousarray(np.transpose(self.target_texture, (2, 0, 1))),
            'input': np.ascontiguousarray(np.transpose(self.partial_texture, (2, 0, 1))),
            'mask_texture': self.mask_texture[np.newaxis, :, :].astype(np.float32),
            'mask_missing': self.partial_mask[np.newaxis, :, :].astype(np.float32),
            'input_patch': np.ascontiguousarray(np.transpose(patch, (2, 0, 1))),
        }

    def denormalize_and_rgb(self, arr):
        return denormalize_and_rgb(arr, self.color_space, self.to_rgb, False)

    def get_patch_from_tensor(self, tensor):
        batch_size = tensor.shape[0]
        patches = []
        for i in range(batch_size):
            idx = random.randint(0, self.non_zero_target[0].shape[0] - 1)
            patches.append(tensor[i: i + 1, :, self.non_zero_target[0][idx]: self.non_zero_target[0][idx] + self.patch_size, self.non_zero_target[1][idx]: self.non_zero_target[1][idx] + self.patch_size])
        return torch.cat(patches, dim=0)

    def visualize_sample_pyplot(self, incomplete, target, mask, missing, patch_input, patch_generated):
        import matplotlib.pyplot as plt
        incomplete = self.denormalize_and_rgb(np.transpose(incomplete, (1, 2, 0)))
        target = self.denormalize_and_rgb(np.transpose(target, (1, 2, 0)))
        mask = mask.squeeze()
        missing = missing.squeeze()
        patch_input = self.denormalize_and_rgb(np.transpose(patch_input, (1, 2, 0)))
        patch_generated = self.denormalize_and_rgb(np.transpose(patch_generated, (1, 2, 0)))
        f, axarr = plt.subplots(1, 8, figsize=(32, 4))
        items = [missing, incomplete, mask, target, patch_input, patch_generated, self.mask_eroded_input, self.mask_eroded_target]
        for i in range(8):
            axarr[i].imshow(items[i])
            axarr[i].axis('off')
        plt.tight_layout()
        plt.show()
