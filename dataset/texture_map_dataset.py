import random
from PIL import Image
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from pathlib import Path
from util.misc import read_list
import numpy as np


class TextureMapDataset(Dataset):

    def __init__(self, config, split):
        self.preload = config.dataset.preload
        self.texture_map_size = config.dataset.texture_map_size
        self.render_size = config.dataset.render_size
        self.views_per_shape = config.dataset.views_per_shape
        self.load_distance_field = 'distance_field' in config.inputs
        self.path_to_dataset = Path(config.dataset.data_dir) / config.dataset.name
        splits_file = Path(config.dataset.data_dir) / 'splits' / config.dataset.name / config.dataset.splits_dir / f'{split}.txt'
        item_list = read_list(splits_file)
        # sanity filter
        self.items = [x for x in item_list if (self.path_to_dataset / x / 'surface_texture.png').exists()]
        self.preload_dict = {}
        if self.preload:
            for index in tqdm(range(len(self.items)), desc='preload_texmap_data'):
                df, texture, normal, noc, mask_texture = self.load_view_independent_data_from_disk(index)
                render_list, mask_render_list = [], []
                for view_index in range(self.views_per_shape):
                    render, mask_render = self.load_view_dependent_data_from_disk(index, view_index)
                    render_list.append(render)
                    mask_render_list.append(mask_render)
                self.preload_dict[self.items[index]] = {
                    'df': df,
                    'texture': texture,
                    'normal': normal,
                    'noc': noc,
                    'mask_texture': mask_texture,
                    'render': render_list,
                    'mask_render': mask_render_list
                }

        if config.dataset.splits_dir.startswith('overfit'):
            multiplier = 240 if split == 'train' else 24
            self.items = self.items * multiplier

    def load_view_independent_data_from_disk(self, item_index):
        item = self.items[item_index]
        df_path = self.path_to_dataset / item / "shape_df.npz"
        texture_path = self.path_to_dataset / item / "surface_texture.png"
        normal_path = self.path_to_dataset / item / "surface_normals.png"
        noc_path = self.path_to_dataset / item / "noc.png"
        df = []
        if self.load_distance_field:
            df = np.load(df_path)['arr'].astype(np.float32)[np.newaxis, :, :, :]
        with Image.open(texture_path) as texture_im:
            texture = TextureMapDataset.process_to_padded_thumbnail(texture_im, self.texture_map_size)
        with Image.open(normal_path) as normal_im:
            normal = TextureMapDataset.process_to_padded_thumbnail(normal_im, self.texture_map_size)
        with Image.open(noc_path) as noc_im:
            noc = TextureMapDataset.process_to_padded_thumbnail(noc_im, self.texture_map_size)
            mask_texture = np.logical_not(np.logical_and(np.logical_and(noc[:, :, 0] == 0, noc[:, :, 1] == 0), noc[:, :, 2] == 0))
        return df, np.ascontiguousarray(np.transpose(texture, (2, 0, 1))), np.ascontiguousarray(np.transpose(normal, (2, 0, 1))), np.ascontiguousarray(np.transpose(noc, (2, 0, 1))), mask_texture[np.newaxis, :, :]

    def load_view_dependent_data_from_disk(self, item_index, view_index):
        item = self.items[item_index]
        image_path = self.path_to_dataset / item / f"rgb_{view_index:03d}.png"
        mask_path = self.path_to_dataset / item / f"silhoutte_{view_index:03d}.png"
        with Image.open(image_path) as render_im:
            render = np.array(render_im).astype(np.float32)
        with Image.open(mask_path) as mask_im:
            mask_render = np.array(mask_im)
            mask_render = np.logical_and(np.logical_and(mask_render[:, :, 0] < 50, mask_render[:, :, 1] < 50), mask_render[:, :, 2] < 50)
        render[~mask_render, :] = 0
        render = render[:, :, :3]
        return np.ascontiguousarray(np.transpose(render, (2, 0, 1))), mask_render[np.newaxis, :, :]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        view_index = random.randint(0, self.views_per_shape - 1)
        if self.preload:
            df, texture, normal, noc, mask_texture = self.preload_dict[item]['df'], self.preload_dict[item]['texture'], self.preload_dict[item]['normal'], self.preload_dict[item]['noc'], self.preload_dict[item]['mask_texture']
            render_list, mask_render_list = self.preload_dict[item]['render'], self.preload_dict[item]['mask_render']
            render = render_list[view_index]
            mask_render = mask_render_list[view_index]
        else:
            df, texture, normal, noc, mask_texture = self.load_view_independent_data_from_disk(index)
            render, mask_render = self.load_view_dependent_data_from_disk(index, view_index)
        return {
            'name': f'{item}',
            'view_index': view_index,
            'df': df,
            'texture': texture,
            'normal': normal,
            'noc': noc,
            'mask_texture': mask_texture.astype(np.float32),
            'render': render,
            'mask_render': mask_render.astype(np.float32)
        }

    @staticmethod
    def apply_batch_transforms(batch):
        batch['texture'] = batch['texture'] / 255 - 0.5
        batch['normal'] = batch['normal'] / 255 - 0.5
        batch['noc'] = batch['noc'] / 255 - 0.5
        batch['render'] = batch['render'] / 255 - 0.5

    @staticmethod
    def move_batch_to_gpu(batch, device):
        batch['texture'] = batch['texture'].to(device)
        batch['normal'] = batch['normal'].to(device)
        batch['noc'] = batch['noc'].to(device)
        batch['mask_texture'] = batch['mask_texture'].to(device)
        batch['render'] = batch['render'].to(device)
        batch['mask_render'] = batch['mask_render'].to(device)
        if type(batch['df']) != list:
            batch['df'] = batch['df'].to(device)

    @staticmethod
    def visualize_sample_pyplot(texture, normal, noc, mask_texture, render, mask_render):
        import matplotlib.pyplot as plt
        texture = np.transpose(texture, (1, 2, 0))
        normal = np.transpose(normal, (1, 2, 0))
        noc = np.transpose(noc, (1, 2, 0))
        render = np.transpose(render, (1, 2, 0))
        mask_render = mask_render.squeeze()
        mask_texture = mask_texture.squeeze()
        f, axarr = plt.subplots(2, 3, figsize=(4, 6))
        axarr[0, 0].imshow(texture + 0.5)
        axarr[0, 1].imshow(normal + 0.5)
        axarr[0, 2].imshow(render + 0.5)
        axarr[1, 0].imshow(noc + 0.5)
        axarr[1, 1].imshow(mask_texture + 0.5)
        axarr[1, 2].imshow(mask_render + 0.5)
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
