import random

import torch
from PIL import Image
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from pathlib import Path

from model.fold import Fold2D
from util.misc import read_list, move_batch_to_gpu, apply_batch_color_transform_and_normalization, normalize_tensor_color
import numpy as np
from colorspacious import cspace_convert


class TextureMapDataset(Dataset):

    def __init__(self, config, split, preload_dict):
        self.preload = config.dataset.preload
        self.texture_map_size = config.dataset.texture_map_size
        self.render_size = config.dataset.render_size
        self.views_per_shape = config.dataset.views_per_shape
        self.color_space = config.dataset.color_space
        self.preload_dict = preload_dict
        self.from_rgb, self.to_rgb = self.convert_cspace(config.dataset.color_space)
        self.load_distance_field = 'distance_field' in config.inputs
        self.path_to_dataset = Path(config.dataset.data_dir) / config.dataset.name
        self.all_view_indexing_mode = False
        splits_file = Path(config.dataset.data_dir) / 'splits' / config.dataset.name / config.dataset.splits_dir / f'{split}.txt'
        item_list = read_list(splits_file)
        # sanity filter
        self.items = [x for x in item_list if (self.path_to_dataset / x / 'surface_texture.png').exists()]
        self.load_retrievals = 'retrievals' in config.inputs
        if self.load_retrievals:
            self.retrieval_dir = Path(config.dictionary.retrieval_dir)
        if self.preload:
            for index in tqdm(range(len(self.items)), desc='preload_texmap_data'):
                if self.items[index] not in preload_dict:
                    df, texture, normal, noc, mask_texture = self.load_view_independent_data_from_disk(index)
                    render_list, noc_render_list, mask_render_list, partial_texture_list, retrievals_list = [], [], [], [], []
                    for view_index in range(self.views_per_shape):
                        render, noc_render, mask_render, partial_texture, retrievals = self.load_view_dependent_data_from_disk(index, view_index)
                        render_list.append(render)
                        noc_render_list.append(noc_render)
                        mask_render_list.append(mask_render)
                        partial_texture_list.append(partial_texture)
                        retrievals_list.append(retrievals)
                    self.preload_dict[self.items[index]] = {
                        'df': df,
                        'texture': texture,
                        'normal': normal,
                        'noc': noc,
                        'noc_render': noc_render_list,
                        'mask_texture': mask_texture,
                        'render': render_list,
                        'mask_render': mask_render_list,
                        'partial_texture': partial_texture_list,
                        'retrievals': retrievals_list
                    }
        elif hasattr(config, 'dictionary') and config.dictionary is not None and not hasattr(config.dictionary, 'retrieval_dir'):  # retrieval task
            for index in tqdm(range(len(self.items)), desc=f'preload_texmaps_{split}'):
                if self.items[index] not in self.preload_dict:
                    self.preload_dict[self.items[index]] = {
                        'texture': self.get_texture(self.items[index])
                    }

        if config.dataset.splits_dir.startswith('overfit'):
            multiplier = 240 if split == 'train' else 24
            self.items = self.items * multiplier

    def load_view_independent_data_from_disk(self, item_index):
        item, _ = self.get_item_and_view_idx(item_index)
        df_path = self.path_to_dataset / item / "shape_df.npz"
        texture_path = self.path_to_dataset / item / "surface_texture.png"
        normal_path = self.path_to_dataset / item / "surface_normals.png"
        noc_path = self.path_to_dataset / item / "noc.png"
        df = []
        if self.load_distance_field:
            df = np.load(df_path)['arr'].astype(np.float32)[np.newaxis, :, :, :]
        with Image.open(texture_path) as texture_im:
            texture = self.from_rgb(TextureMapDataset.process_to_padded_thumbnail(texture_im, self.texture_map_size)).astype(np.float32)
        with Image.open(normal_path) as normal_im:
            normal = TextureMapDataset.process_to_padded_thumbnail(normal_im, self.texture_map_size)
        with Image.open(noc_path) as noc_im:
            noc = TextureMapDataset.process_to_padded_thumbnail(noc_im, self.texture_map_size)
            mask_texture = np.logical_not(np.logical_and(np.logical_and(noc[:, :, 0] == 0, noc[:, :, 1] == 0), noc[:, :, 2] == 0))
        return df, np.ascontiguousarray(np.transpose(texture, (2, 0, 1))), np.ascontiguousarray(np.transpose(normal, (2, 0, 1))), np.ascontiguousarray(np.transpose(noc, (2, 0, 1))), mask_texture[np.newaxis, :, :]

    def load_view_dependent_data_from_disk(self, item_index, forced_view_index=None):
        item, view_index = self.get_item_and_view_idx(item_index)
        view_index = view_index if forced_view_index is None else forced_view_index
        image_path = self.path_to_dataset / item / f"rgb_{view_index:03d}.png"
        noc_render_path = self.path_to_dataset / item / f"noc_render_{view_index:03d}.png"
        mask_path = self.path_to_dataset / item / f"silhoutte_{view_index:03d}.png"
        partial_texture_path = self.path_to_dataset / item / f"inv_partial_texture_{view_index:03d}.png"
        retrieval = []
        if self.load_retrievals:
            retrieval = np.load(self.retrieval_dir / 'compose' / f'{item}__{view_index:02d}.npz')["arr_0"]
        with Image.open(image_path) as render_im:
            render = self.from_rgb(np.array(render_im)).astype(np.float32)
        with Image.open(noc_render_path) as render_noc:
            noc_render = np.array(render_noc).astype(np.float32)
        with Image.open(partial_texture_path) as partial_tex:
            partial_texture = self.from_rgb(TextureMapDataset.process_to_padded_thumbnail(partial_tex, self.texture_map_size)).astype(np.float32)
        with Image.open(mask_path) as mask_im:
            mask_render = np.array(mask_im)
            mask_render = np.logical_and(np.logical_and(mask_render[:, :, 0] < 50, mask_render[:, :, 1] < 50), mask_render[:, :, 2] < 50)
        render[~mask_render, :] = 0
        noc_render[~mask_render, :] = 0
        render = render[:, :, :3]
        noc_render = noc_render[:, :, :3]
        return np.ascontiguousarray(np.transpose(render, (2, 0, 1))), np.ascontiguousarray(np.transpose(noc_render, (2, 0, 1))), mask_render[np.newaxis, :, :], np.ascontiguousarray(np.transpose(partial_texture, (2, 0, 1))), retrieval

    def __getitem__(self, index):
        item, view_index = self.get_item_and_view_idx(index)
        if self.preload:
            df, texture, normal, noc, mask_texture = self.preload_dict[item]['df'], self.preload_dict[item]['texture'], self.preload_dict[item]['normal'], self.preload_dict[item]['noc'], self.preload_dict[item]['mask_texture']
            render_list, noc_render_list, mask_render_list, partial_texture_list, retrieval_list = \
                self.preload_dict[item]['render'], self.preload_dict[item]['noc_render'], self.preload_dict[item]['mask_render'], self.preload_dict[item]['partial_texture'], self.preload_dict[item]['retrievals']
            render = render_list[view_index]
            noc_render = noc_render_list[view_index]
            mask_render = mask_render_list[view_index]
            partial_texture = partial_texture_list[view_index]
            retrievals = retrieval_list[view_index]
        else:
            df, texture, normal, noc, mask_texture = self.load_view_independent_data_from_disk(index)
            render, noc_render, mask_render, partial_texture, retrievals = self.load_view_dependent_data_from_disk(index)

        return {
            'name': f'{item}',
            'view_index': view_index,
            'df': df,
            'retrievals': retrievals,
            'texture': texture,
            'normal': normal,
            'noc': noc,
            'noc_render': noc_render,
            'mask_texture': mask_texture.astype(np.float32),
            'render': render,
            'mask_render': mask_render.astype(np.float32),
            'partial_texture': partial_texture
        }

    def __len__(self):
        return len(self.items) * (self.views_per_shape if self.all_view_indexing_mode else 1)

    def apply_batch_transforms(self, batch):
        items_color = ['texture', 'render', 'partial_texture']
        items_non_color = ['normal', 'noc', 'noc_render']
        apply_batch_color_transform_and_normalization(batch, items_color, items_non_color, self.color_space)
        batch['texture'] = self.apply_mask_texture(batch['texture'], batch['mask_texture'])
        # rand_index = random.randint(0, 7)
        # batch['retrievals'][:, rand_index, :, :, :] = batch['texture']

    @staticmethod
    def apply_mask_texture(texture, mask):
        expanded_mask = mask.expand(-1, texture.shape[1], -1, -1)
        return texture * expanded_mask

    def convert_data_for_visualization(self, colored_items, non_colored_items, masks):
        for i in range(len(colored_items)):
            colored_items[i] = np.transpose(colored_items[i], (1, 2, 0))
            colored_items[i] = self.get_colored_data_for_visualization(colored_items[i])
        for i in range(len(non_colored_items)):
            non_colored_items[i] = np.transpose(non_colored_items[i], (1, 2, 0))
            non_colored_items[i] = non_colored_items[i] + 0.5
        for i in range(len(masks)):
            masks[i] = masks[i].squeeze()
        return colored_items, non_colored_items, masks

    def get_colored_data_for_visualization(self, item):
        if self.color_space == 'rgb':
            item = item + 0.5
        elif self.color_space == 'lab':
            item[:, :, 0] = (item[:, :, 0] + 0.5) * 100
            item[:, :, 1] = np.clip(item[:, :, 1] * 256, -128, 127)
            item[:, :, 2] = np.clip(item[:, :, 2] * 256, -128, 127)
            # item[:, :, 1:] = 0
            item = np.clip(self.to_rgb(item), 0, 255) / 255
        return item

    @staticmethod
    def move_batch_to_gpu(batch, device):
        keys = ['texture', 'normal', 'noc', 'noc_render', 'mask_texture', 'render', 'mask_render', 'partial_texture']
        if type(batch['df']) != list:
            keys.append('df')
        if type(batch['retrievals']) != list:
            keys.append('retrievals')
        move_batch_to_gpu(batch, device, keys)

    def visualize_sample_pyplot(self, texture, normal, noc, mask_texture, render, noc_render, mask_render, partial_texture, sampled_patches):
        import matplotlib.pyplot as plt
        [texture, render, partial_texture], [normal, noc, noc_render], [mask_texture, mask_render] = self.convert_data_for_visualization([texture, render, partial_texture], [normal, noc, noc_render], [mask_texture, mask_render])
        sampled_patches, _, _ = self.convert_data_for_visualization([sampled_patches[i] for i in range(sampled_patches.shape[0])], [], [])
        rows, cols = 3, 4
        f, axarr = plt.subplots(3, 4, figsize=(24, 4))
        items = [[texture, normal, render, noc_render], [noc, mask_texture, mask_render, partial_texture], sampled_patches[:4]]
        for i in range(rows):
            for j in range(cols):
                axarr[i, j].imshow(items[i][j])
                axarr[i, j].axis('off')
        plt.tight_layout()
        plt.show()

    @staticmethod
    def process_to_padded_thumbnail(image, size):
        image.thumbnail((size, size))
        thumbnail = np.zeros((size, size, 3), dtype=np.float32)
        image_array = np.array(image).astype(np.float32)
        if image_array.shape[2] == 2:
            thumbnail[:image_array.shape[0], :image_array.shape[1], 0] = image_array[:, :, 0]
            thumbnail[:image_array.shape[0], :image_array.shape[1], 1] = image_array[:, :, 0]
            thumbnail[:image_array.shape[0], :image_array.shape[1], 2] = image_array[:, :, 0]
        else:
            thumbnail[:image_array.shape[0], :image_array.shape[1], :] = image_array[:, :, :3]
        return thumbnail

    @staticmethod
    def convert_cspace(cspace):
        to_rgb = lambda x: x
        from_rgb = lambda x: x
        if cspace == 'lab':
            to_rgb = lambda x: cspace_convert(x, 'CIELab', 'sRGB255')
            from_rgb = lambda x: cspace_convert(x[:, :, :3], 'sRGB255', 'CIELab')
        return from_rgb, to_rgb

    @staticmethod
    def get_valid_sampling_area(mask, patch_size):
        valid_area = mask.clone()
        mask_shifted_dim_2 = torch.zeros_like(mask)
        mask_shifted_dim_2[:, :, :-patch_size, :] = mask[:, :, patch_size:, :]
        zero_dim_2 = torch.logical_and(torch.logical_not(mask_shifted_dim_2), mask)
        mask_shifted_dim_3 = torch.zeros_like(mask)
        mask_shifted_dim_3[:, :, :, :-patch_size] = mask[:, :, :, patch_size:]
        zero_dim_3 = torch.logical_and(torch.logical_not(mask_shifted_dim_3), mask)
        mask_shifted_dim_23 = torch.zeros_like(mask)
        mask_shifted_dim_23[:, :, :-patch_size, :-patch_size] = mask[:, :, patch_size:, patch_size:]
        zero_dim_23 = torch.logical_and(torch.logical_not(mask_shifted_dim_23), mask)
        valid_area[zero_dim_2] = 0
        valid_area[zero_dim_3] = 0
        valid_area[zero_dim_23] = 0
        return valid_area

    @staticmethod
    def sample_patches(mask, patch_size, num_patches, *tensors):
        valid_area = TextureMapDataset.get_valid_sampling_area(mask, patch_size)
        batch_size = tensors[0].shape[0]
        all_samples = [[[] for _0 in range(batch_size)] for _1 in range(len(tensors))]
        for b in range(batch_size):
            samples = torch.where(valid_area[b, 0, :, :])
            for tid in range(len(tensors)):
                if num_patches == -1:
                    indices = list(range(samples[0].shape[0]))
                else:
                    indices = random.sample(list(range(samples[0].shape[0])), min(num_patches, samples[0].shape[0]))
                for k in indices:
                    sampled_generated = tensors[tid][b: b + 1, :, samples[0][k]: samples[0][k] + patch_size, samples[1][k]: samples[1][k] + patch_size]
                    all_samples[tid][b].append(sampled_generated.cpu())
        for tid in range(len(tensors)):
            common_num_patches = min([len(all_samples[tid][b]) for b in range(batch_size)])
            for b in range(batch_size):
                all_samples[tid][b] = all_samples[tid][b][:common_num_patches]
        return [torch.cat([torch.cat(all_samples[tid][b], dim=0).unsqueeze(0) for b in range(batch_size)], dim=0) for tid in range(len(tensors))]

    def sample_patches_for_ptexture(self, partial_texture_name, patch_size, num_textures):
        fold2d = Fold2D(self.texture_map_size // patch_size, patch_size, 3)
        texture = partial_texture_name.split("__")[0]
        view_idx = int(partial_texture_name.split("__")[1])
        partial_texture = torch.from_numpy(self.get_partial_texture(texture, view_idx)).unsqueeze(0)
        partial_texture = normalize_tensor_color(partial_texture, self.color_space)
        with Image.open(self.path_to_dataset / texture / f"inv_partial_mask_{view_idx:03d}.png") as mask_im:
            mask_im.thumbnail((self.texture_map_size, self.texture_map_size))
            partial_mask = torch.from_numpy(np.array(mask_im)[:, :, 0] > 0).unsqueeze(0).unsqueeze(0)
        num_patches = (self.texture_map_size // patch_size) ** 2
        all_samples = []
        for i in range(num_textures):
            all_samples.append(fold2d(TextureMapDataset.sample_patches(partial_mask, patch_size, num_patches, partial_texture)[0]))
        return torch.cat(all_samples, dim=0)

    def get_item_and_view_idx(self, index):
        if self.all_view_indexing_mode:
            item = self.items[index // self.views_per_shape]
            view_index = index % self.views_per_shape
        else:
            item = self.items[index]
            view_index = random.randint(0, self.views_per_shape - 1)
        return item, view_index

    def set_all_view_indexing(self, value):
        self.all_view_indexing_mode = value

    def get_texture(self, name):
        if name in self.preload_dict:
            texture = self.preload_dict[name]['texture']
            return texture.copy()
        texture_path = self.path_to_dataset / name / "surface_texture.png"
        with Image.open(texture_path) as texture_im:
            texture = self.from_rgb(TextureMapDataset.process_to_padded_thumbnail(texture_im, self.texture_map_size)).astype(np.float32)
        return np.ascontiguousarray(np.transpose(texture, (2, 0, 1)))

    def get_mask(self, name):
        noc_path = self.path_to_dataset / name / "noc.png"
        with Image.open(noc_path) as noc_im:
            noc = TextureMapDataset.process_to_padded_thumbnail(noc_im, self.texture_map_size)
            mask_texture = np.logical_not(np.logical_and(np.logical_and(noc[:, :, 0] == 0, noc[:, :, 1] == 0), noc[:, :, 2] == 0))
        return mask_texture[np.newaxis, :, :]

    def get_partial_texture(self, name, view_index):
        partial_texture_path = self.path_to_dataset / name / f"inv_partial_texture_{view_index:03d}.png"
        with Image.open(partial_texture_path) as partial_tex:
            partial_texture = self.from_rgb(TextureMapDataset.process_to_padded_thumbnail(partial_tex, self.texture_map_size)).astype(np.float32)
        return np.ascontiguousarray(np.transpose(partial_texture, (2, 0, 1)))
