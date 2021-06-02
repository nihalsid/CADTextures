import torch
from pathlib import Path
import numpy as np
import random
from PIL import Image
from tqdm import tqdm
import math

from dataset.texture_map_dataset import TextureMapDataset
from model.attention import Unfold2D, Fold2D
from util.misc import read_list, move_batch_to_gpu, apply_batch_color_transform_and_normalization, denormalize_and_rgb, normalize_tensor_color


class TextureEnd2EndDataset(torch.utils.data.Dataset):

    def __init__(self, config, split, preload_dict, single_view=False, load_database=True):
        super().__init__()
        self.load_database = load_database
        self.preload = config.dataset.preload
        self.preload_dict = preload_dict
        self.views_per_shape = 1 if (split == 'val' or split == 'val_vis' or split == 'train_vis' or single_view) else config.dataset.views_per_shape
        self.unfold = Unfold2D(config.dictionary.patch_size, 3)
        self.texture_map_size = config.dataset.texture_map_size
        self.color_space = config.dataset.color_space
        self.from_rgb, self.to_rgb = TextureMapDataset.convert_cspace(config.dataset.color_space)
        self.path_to_dataset = Path(config.dataset.data_dir) / config.dataset.name
        self.num_database_textures = config.dataset.num_database_textures
        splits_file = Path(config.dataset.data_dir) / 'splits' / config.dataset.name / config.dataset.splits_dir / f'{split}.txt'
        train_splits_file = Path(config.dataset.data_dir) / 'splits' / config.dataset.name / config.dataset.splits_dir / f'train.txt'
        item_list = read_list(splits_file)
        train_item_list = read_list(train_splits_file)
        # sanity filter
        self.items = [x for x in item_list if (self.path_to_dataset / x / 'surface_texture.png').exists()]
        self.train_items = [x for x in train_item_list if (self.path_to_dataset / x / 'surface_texture.png').exists()]
        if self.preload:
            for index in tqdm(range(len(self.items)), desc='preload_texmap_data'):
                if self.items[index] not in preload_dict:
                    texture, mask_texture = self.load_view_independent_data_from_disk(self.items[index])
                    partial_texture_list = []
                    for view_index in range(self.views_per_shape):
                        partial_texture = self.load_view_dependent_data_from_disk(self.items[index], view_index)
                        partial_texture_list.append(partial_texture)
                    self.preload_dict[self.items[index]] = {
                        'texture': texture,
                        'mask_texture': mask_texture,
                        'partial_texture': partial_texture_list,
                    }
        if config.dataset.splits_dir.startswith('overfit'):
            multiplier = 240 if split == 'train' else 4
            self.items = self.items * multiplier
            self.train_items = self.train_items * 240

    def load_view_independent_data_from_disk(self, item):
        texture_path = self.path_to_dataset / item / "surface_texture.png"
        noc_path = self.path_to_dataset / item / "noc.png"
        with Image.open(texture_path) as texture_im:
            texture = self.from_rgb(TextureMapDataset.process_to_padded_thumbnail(texture_im, self.texture_map_size)).astype(np.float32)
        with Image.open(noc_path) as noc_im:
            noc = TextureMapDataset.process_to_padded_thumbnail(noc_im, self.texture_map_size)
            mask_texture = np.logical_not(np.logical_and(np.logical_and(noc[:, :, 0] == 0, noc[:, :, 1] == 0), noc[:, :, 2] == 0))
        return np.ascontiguousarray(np.transpose(texture, (2, 0, 1))), mask_texture[np.newaxis, :, :]

    def load_view_dependent_data_from_disk(self, item, view_index):
        partial_texture_path = self.path_to_dataset / item / f"inv_partial_texture_{view_index:03d}.png"
        with Image.open(partial_texture_path) as partial_tex:
            partial_texture = self.from_rgb(TextureMapDataset.process_to_padded_thumbnail(partial_tex, self.texture_map_size)).astype(np.float32)
        return np.ascontiguousarray(np.transpose(partial_texture, (2, 0, 1)))

    def apply_batch_transforms(self, batch, texture_masking=True):
        normalization_keys = ['texture', 'partial_texture']
        if not type(batch['database_textures']) == list:
            batch['database_textures'] = self.unfold(batch['database_textures'].reshape([-1, batch['database_textures'].shape[2], batch['database_textures'].shape[3], batch['database_textures'].shape[4]]))
            normalization_keys.append('database_textures')
        apply_batch_color_transform_and_normalization(batch, normalization_keys, [], self.color_space)
        if texture_masking:
            batch['texture'] = TextureMapDataset.apply_mask_texture(batch['texture'], batch['mask_texture'])
        batch['partial_texture'] = TextureMapDataset.apply_mask_texture(batch['partial_texture'], batch['mask_texture'])

    @staticmethod
    def move_batch_to_gpu(batch, device):
        move_keys = ['texture', 'partial_texture', 'mask_texture']
        if not type(batch['database_textures']) == list:
            move_keys.append('database_textures')
        move_batch_to_gpu(batch, device, move_keys)

    def denormalize_and_rgb(self, arr):
        return denormalize_and_rgb(arr, self.color_space, self.to_rgb, False)

    def __getitem__(self, index):
        view_index = index % self.views_per_shape
        item = self.items[index // self.views_per_shape]
        database_textures = []
        if self.preload:
            texture, mask_texture, partial_texture = self.preload_dict[item]['texture'], self.preload_dict[item]['mask_texture'], self.preload_dict[item]['partial_texture'][view_index]
            if self.load_database:
                database_texture_names = self.items[:self.num_database_textures] # random.sample(self.items, self.num_database_textures)
                for tex in database_texture_names:
                    database_textures.append(self.preload_dict[tex]['texture'][np.newaxis, :, :, :])
                database_textures = np.concatenate(database_textures, axis=0)
        else:
            texture, mask_texture = self.load_view_independent_data_from_disk(item)
            partial_texture = self.load_view_dependent_data_from_disk(item, view_index)
            if self.load_database:
                database_texture_names = random.sample(self.items, self.num_database_textures)
                for tex in database_texture_names:
                    database_textures.append(self.load_view_independent_data_from_disk(tex)[0][np.newaxis, :, :, :])
                database_textures = np.concatenate(database_textures, axis=0)
        return {
            'name': f'{item}',
            'view_index': view_index,
            'texture': texture,
            'database_textures': database_textures,
            'mask_texture': mask_texture.astype(np.float32),
            'partial_texture': partial_texture
        }

    def visualize_sample_pyplot(self, incomplete, target, mask, database):
        import matplotlib.pyplot as plt
        incomplete = self.denormalize_and_rgb(np.transpose(incomplete, (1, 2, 0)))
        target = self.denormalize_and_rgb(np.transpose(target, (1, 2, 0)))
        mask = mask.squeeze()
        database = database.copy()
        database_patches = [self.denormalize_and_rgb(np.transpose(database[i, :, :, :], (1, 2, 0))) for i in range(database.shape[0])]
        sampled_patches = random.sample(range(len(database_patches)), 4 * 4)
        f, axarr = plt.subplots(5, 4, figsize=(16, 20))
        items = [incomplete, mask, target, np.zeros([8, 8])]
        for i in range(4):
            axarr[0, i].imshow(items[i])
            axarr[0, i].axis('off')
        for i in range(1, 5):
            for j in range(0, 4):
                axarr[i, j].imshow(database_patches[sampled_patches[(i - 1) * 4 + j]])
                axarr[i, j].axis('off')
        plt.tight_layout()
        plt.show()

    def visualize_texture_batch(self, texture_batch, outpath):
        import matplotlib.pyplot as plt
        texture_batch = texture_batch.copy()
        texture_batch_items_row_0 = [self.denormalize_and_rgb(np.transpose(texture_batch[i, :, :, :], (1, 2, 0))) for i in range(texture_batch.shape[0] // 2)]
        texture_batch_items_row_1 = [self.denormalize_and_rgb(np.transpose(texture_batch[i, :, :, :], (1, 2, 0))) for i in range(texture_batch.shape[0]//2, texture_batch.shape[0])]
        f, axarr = plt.subplots(2, len(texture_batch_items_row_0), figsize=(4 * len(texture_batch_items_row_0), 8))
        for i in range(len(texture_batch_items_row_0)):
            axarr[0, i].imshow(texture_batch_items_row_0[i])
            axarr[0, i].axis('off')
            axarr[1, i].imshow(texture_batch_items_row_1[i])
            axarr[1, i].axis('off')
        plt.tight_layout()
        plt.savefig(outpath, bbox_inches='tight', dpi=240)
        plt.close()

    def __len__(self):
        return len(self.items) * self.views_per_shape

    def get_all_texture_patch_codes(self, fenc_target, device, batch_size):
        assert self.preload
        codes = []
        num_database_textures = self.num_database_textures * batch_size
        for i in range(math.ceil(len(self.train_items) / num_database_textures)):
            batch = dict({'database_textures': []})
            for tex in self.train_items[i * num_database_textures: (i + 1) * num_database_textures]:
                batch['database_textures'].append(self.preload_dict[tex]['texture'][np.newaxis, :, :, :].copy())
            batch['database_textures'] = torch.from_numpy(np.concatenate(batch['database_textures'], axis=0)).to(device)
            batch['database_textures'] = self.unfold(batch['database_textures'])
            apply_batch_color_transform_and_normalization(batch, ['database_textures'], [], self.color_space)
            codes.append(torch.nn.functional.normalize(fenc_target(batch['database_textures']), dim=1).cpu())
        return torch.cat(codes, dim=0)

    def get_patches_with_indices(self, selections):
        assert self.preload
        num_patch_x = self.texture_map_size // self.unfold.patch_extent
        patches = torch.zeros((selections.shape[0], 3, self.unfold.patch_extent, self.unfold.patch_extent))
        for i in range(selections.shape[0]):
            texture_index = selections[i] // (num_patch_x * num_patch_x)
            patch_index = selections[i] % (num_patch_x * num_patch_x)
            row = patch_index // num_patch_x
            col = patch_index % num_patch_x
            patch = self.preload_dict[self.train_items[texture_index]]['texture'][:, row * self.unfold.patch_extent: (row + 1) * self.unfold.patch_extent, col * self.unfold.patch_extent: (col + 1) * self.unfold.patch_extent].copy()
            patches[i, :] = normalize_tensor_color(torch.from_numpy(patch).unsqueeze(0), self.color_space).squeeze(0)
        return patches
