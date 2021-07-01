from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from tqdm import tqdm

from dataset.texture_map_dataset import TextureMapDataset
from util.misc import read_list, move_batch_to_gpu


class ImplicitDataset(Dataset):

    def __init__(self, config, split, preload_dict, single_view=False):
        super().__init__()
        self.preload = config.dataset.preload
        self.preload_dict = preload_dict
        self.views_per_shape = 1 if (split == 'val' or split == 'val_vis' or split == 'train_vis' or single_view) else config.dataset.views_per_shape
        self.texture_map_size = config.dataset.texture_map_size
        self.path_to_dataset = Path(config.dataset.data_dir) / config.dataset.name
        self.sample_points_per_object = config.num_samples
        splits_file = Path(config.dataset.data_dir) / 'splits' / config.dataset.name / config.dataset.splits_dir / f'{split}.txt'
        item_list = read_list(splits_file)
        self.items = [x for x in item_list if (self.path_to_dataset / x / 'surface_texture.png').exists()]
        if self.preload:
            for index in tqdm(range(len(self.items)), desc='preload_texmap_data'):
                if self.items[index] not in preload_dict:
                    partial_texture_list, missing_mask_list = [], []
                    npz = np.load(self.path_to_dataset / self.items[index] / f"implicit_samples.npz")
                    grid_coords, values = npz['points'].astype(np.float32) * 2, npz['values'].astype(np.float32) / 255 - 0.5
                    texture_mask = self.load_mask_texture(self.items[index])
                    for view_index in range(self.views_per_shape):
                        partial_texture, missing_mask = self.load_view_dependent_data_from_disk(self.items[index], view_index)
                        partial_texture_list.append(partial_texture)
                        missing_mask_list.append(missing_mask)
                    self.preload_dict[self.items[index]] = {
                        'grid_coords': grid_coords,
                        'values': values,
                        'partial_texture': partial_texture_list,
                        'missing_mask': missing_mask_list,
                        'texture_mask': texture_mask
                    }
        if config.dataset.splits_dir.startswith('overfit'):
            multiplier = 240 if split == 'train' else 2
            self.items = self.items * multiplier

    def load_mask_texture(self, item):
        noc_path = self.path_to_dataset / item / "noc.png"
        with Image.open(noc_path) as noc_im:
            noc = TextureMapDataset.process_to_padded_thumbnail(noc_im, self.texture_map_size)
            mask_texture = np.logical_not(np.logical_and(np.logical_and(noc[:, :, 0] == 0, noc[:, :, 1] == 0), noc[:, :, 2] == 0))
        return mask_texture[np.newaxis, :, :]

    def load_view_dependent_data_from_disk(self, item, view_index):
        missing_mask_path = self.path_to_dataset / item / f"inv_partial_mask_{view_index:03d}.png"
        partial_texture_path = self.path_to_dataset / item / f"inv_partial_texture_{view_index:03d}.png"
        with Image.open(missing_mask_path) as mask_im:
            mask_im.thumbnail((self.texture_map_size, self.texture_map_size))
            missing_mask = np.array(mask_im)[:, :, 0] == 0
            missing_mask = missing_mask[np.newaxis, :, :].astype(np.float32)
        with Image.open(partial_texture_path) as partial_tex:
            partial_texture = TextureMapDataset.process_to_padded_thumbnail(partial_tex, self.texture_map_size).astype(np.float32)
        return np.ascontiguousarray(np.transpose(partial_texture, (2, 0, 1))), missing_mask

    def __len__(self):
        return len(self.items) * self.views_per_shape

    def __getitem__(self, index):
        view_index = index % self.views_per_shape
        item = self.items[index // self.views_per_shape]
        if self.preload:
            grid_coords, values, partial_texture = self.preload_dict[item]['grid_coords'], self.preload_dict[item]['values'], self.preload_dict[item]['partial_texture'][view_index]
            missing_mask, texture_mask = self.preload_dict[item]['missing_mask'][view_index], self.preload_dict[item]['texture_mask']
        else:
            partial_texture, missing_mask = self.load_view_dependent_data_from_disk(item, view_index)
            npz = np.load(self.path_to_dataset / item / f"implicit_samples.npz")
            grid_coords, values = npz['points'].astype(np.float32) * 2, npz['values'].astype(np.float32) / 255.0 - 0.5
            texture_mask = self.load_mask_texture(item)
        subsample_indices = np.random.randint(0, grid_coords.shape[0], self.sample_points_per_object)
        grid_coords = grid_coords[subsample_indices]
        values = values[subsample_indices]
        return {
            'name': f'{item}',
            'view_index': view_index,
            'grid_coords': grid_coords,
            'values': values,
            'mask_missing': missing_mask,
            'mask_texture': texture_mask,
            'partial_texture': partial_texture / 255.0 - 0.5
        }

    def get_true_texture(self, names):
        all_textures = []
        for idx, name in enumerate(names):
            texture_path = self.path_to_dataset / name / "surface_texture.png"
            with Image.open(texture_path) as texture_im:
                texture = TextureMapDataset.process_to_padded_thumbnail(texture_im, self.texture_map_size).astype(np.float32) / 255.0 - 0.5
                all_textures.append(texture)
        return all_textures

    @staticmethod
    def visualize_texture_batch(gt, pred, mask, outpath):
        import matplotlib.pyplot as plt
        gt = [(x.copy() + 0.5) * mask[idx, 0, :, :][:, :, np.newaxis] for idx, x in enumerate(gt)]
        pred = [(x.copy() + 0.5) * mask[idx, 0, :, :][:, :, np.newaxis] for idx, x in enumerate(pred)]
        f, axarr = plt.subplots(2, len(gt), figsize=(4 * len(gt), 8))
        for i in range(len(gt)):
            axarr[0, i].imshow(gt[i])
            axarr[0, i].axis('off')
            axarr[1, i].imshow(pred[i])
            axarr[1, i].axis('off')
        plt.tight_layout()
        plt.savefig(outpath, bbox_inches='tight', dpi=240)
        plt.close()

    @staticmethod
    def move_batch_to_gpu(batch, device):
        move_keys = ['grid_coords', 'partial_texture', 'values', 'mask_missing', 'mask_texture']
        move_batch_to_gpu(batch, device, move_keys)
