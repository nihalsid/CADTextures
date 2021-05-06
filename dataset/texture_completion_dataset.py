import torch
from pathlib import Path
import numpy as np
import random
from PIL import Image

from dataset.texture_map_dataset import TextureMapDataset
from util.misc import read_list, move_batch_to_gpu, apply_batch_color_transform_and_normalization


class TextureCompletionDataset(torch.utils.data.Dataset):

    def __init__(self, config, split):
        super().__init__()
        self.views_per_shape = config.dataset.views_per_shape // (4 if (split == 'val_vis' or split == 'train_vis') else 1)
        self.texture_map_size = config.dataset.texture_map_size
        self.color_space = config.dataset.color_space
        self.from_rgb, self.to_rgb = TextureMapDataset.convert_cspace(config.dataset.color_space)
        self.path_to_dataset = Path(config.dataset.data_dir) / config.dataset.name
        splits_file = Path(config.dataset.data_dir) / 'splits' / config.dataset.name / config.dataset.splits_dir / f'{split}.txt'
        item_list = read_list(splits_file)
        # sanity filter
        self.items = [x for x in item_list if (self.path_to_dataset / x / 'texture_complete.png').exists()]
        if config.dataset.splits_dir.startswith('overfit'):
            multiplier = 240 if split == 'train' else 1
            self.items = self.items * multiplier

    def apply_batch_transforms(self, batch):
        apply_batch_color_transform_and_normalization(batch, ['target', 'input'], [], self.color_space)
        batch['target'] = TextureMapDataset.apply_mask_texture(batch['target'], batch['mask'])
        batch['input'] = TextureMapDataset.apply_mask_texture(batch['input'], batch['mask'])

    @staticmethod
    def move_batch_to_gpu(batch, device):
        move_batch_to_gpu(batch, device, ['target', 'input', 'mask', 'missing'])

    def denormalize_and_rgb(self, arr):
        return denormalize_and_rgb(arr, self.color_space, self.to_rgb, False)

    def __getitem__(self, index):
        view_index = index % self.views_per_shape
        item = self.items[index // self.views_per_shape]
        # return mask, incomplete mask, texture, incomplete texture
        mask_path = self.path_to_dataset / item / "mask.png"
        missing_mask_path = self.path_to_dataset / item / f"missing_{view_index:02d}.png"
        missing_texture_path = self.path_to_dataset / item / f"texture_incomplete_{view_index:02d}.png"
        texture_path = self.path_to_dataset / item / "texture_complete.png"
        with Image.open(texture_path) as texture_im:
            texture = self.from_rgb(TextureMapDataset.process_to_padded_thumbnail(texture_im, self.texture_map_size)).astype(np.float32)
        with Image.open(missing_texture_path) as texture_im:
            missing_texture = self.from_rgb(TextureMapDataset.process_to_padded_thumbnail(texture_im, self.texture_map_size)).astype(np.float32)
        with Image.open(mask_path) as mask_im:
            mask_im.thumbnail((self.texture_map_size, self.texture_map_size))
            mask = np.array(mask_im) > 0
        with Image.open(missing_mask_path) as mask_im:
            mask_im.thumbnail((self.texture_map_size, self.texture_map_size))
            missing_mask = np.array(mask_im) > 0
        return {
            'name': f'{item}',
            'view_index': view_index,
            'target': np.ascontiguousarray(np.transpose(texture, (2, 0, 1))),
            'input': np.ascontiguousarray(np.transpose(missing_texture, (2, 0, 1))),
            'mask': mask[np.newaxis, :, :].astype(np.float32),
            'missing': missing_mask[np.newaxis, :, :].astype(np.float32),
        }

    def visualize_sample_pyplot(self, incomplete, target, mask, missing):
        import matplotlib.pyplot as plt
        incomplete = self.denormalize_and_rgb(np.transpose(incomplete, (1, 2, 0)))
        target = self.denormalize_and_rgb(np.transpose(target, (1, 2, 0)))
        mask = mask.squeeze()
        missing = missing.squeeze()
        f, axarr = plt.subplots(1, 4, figsize=(16, 4))
        items = [missing, incomplete, mask, target]
        for i in range(4):
            axarr[i].imshow(items[i])
            axarr[i].axis('off')
        plt.tight_layout()
        plt.show()

    def __len__(self):
        return len(self.items) * self.views_per_shape
