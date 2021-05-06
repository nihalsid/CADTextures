import torch
from pathlib import Path
import numpy as np
import random

from PIL import Image

from dataset.texture_map_dataset import TextureMapDataset
from util.misc import read_list, apply_batch_color_transform_and_normalization, denormalize_and_rgb


class NoiseDataset(torch.utils.data.Dataset):
    def __init__(self, config, z_dim, size):
        super().__init__()
        self.size = size
        self.z_dim = z_dim
        self.color_space = config.dataset.color_space
        self.from_rgb, self.to_rgb = TextureMapDataset.convert_cspace(config.dataset.color_space)
        self.path_to_dataset = Path(config.dataset.data_dir) / config.dataset.name
        splits_file = Path(config.dataset.data_dir) / 'splits' / config.dataset.name / config.dataset.splits_dir / f'train.txt'
        item_list = read_list(splits_file)
        # sanity filter
        self.items = [x for x in item_list if (self.path_to_dataset / x / 'surface_texture.png').exists()]

    def apply_batch_transform(self, batch):
        apply_batch_color_transform_and_normalization(batch, ['target'], [], self.color_space)

    def denormalize_and_rgb(self, arr, only_l):
        return denormalize_and_rgb(arr, self.color_space, self.to_rgb, only_l)

    def __getitem__(self, index):
        random_noise = np.random.normal(0, 1, size=self.z_dim).astype(np.float32)
        random_texture_path = random.choice(self.items)
        with Image.open(self.path_to_dataset / random_texture_path / 'surface_texture.png') as texture_im:
            random_texture = self.from_rgb(TextureMapDataset.process_to_padded_thumbnail(texture_im, 128)).astype(np.float32)
        return {
            "name": f"{index:04d}",
            "input": random_noise,
            "target": np.ascontiguousarray(np.transpose(random_texture, (2, 0, 1)))
        }

    def __len__(self):
        return self.size
