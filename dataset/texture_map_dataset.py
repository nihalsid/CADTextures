import random
from PIL import Image
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from pathlib import Path
from util.misc import read_list
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
        splits_file = Path(config.dataset.data_dir) / 'splits' / config.dataset.name / config.dataset.splits_dir / f'{split}.txt'
        item_list = read_list(splits_file)
        # sanity filter
        self.items = [x for x in item_list if (self.path_to_dataset / x / 'surface_texture.png').exists()]
        if self.preload:
            for index in tqdm(range(len(self.items)), desc='preload_texmap_data'):
                if self.items[index] not in preload_dict:
                    df, texture, normal, noc, mask_texture = self.load_view_independent_data_from_disk(index)
                    render_list, noc_render_list, mask_render_list, partial_texture_list = [], [], [], []
                    for view_index in range(self.views_per_shape):
                        render, noc_render, mask_render, partial_texture = self.load_view_dependent_data_from_disk(index, view_index)
                        render_list.append(render)
                        noc_render_list.append(noc_render)
                        mask_render_list.append(mask_render)
                        partial_texture_list.append(partial_texture)
                    self.preload_dict[self.items[index]] = {
                        'df': df,
                        'texture': texture,
                        'normal': normal,
                        'noc': noc,
                        'noc_render': noc_render_list,
                        'mask_texture': mask_texture,
                        'render': render_list,
                        'mask_render': mask_render_list,
                        'partial_texture': partial_texture_list
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
            texture = self.from_rgb(TextureMapDataset.process_to_padded_thumbnail(texture_im, self.texture_map_size)).astype(np.float32)
        with Image.open(normal_path) as normal_im:
            normal = TextureMapDataset.process_to_padded_thumbnail(normal_im, self.texture_map_size)
        with Image.open(noc_path) as noc_im:
            noc = TextureMapDataset.process_to_padded_thumbnail(noc_im, self.texture_map_size)
            mask_texture = np.logical_not(np.logical_and(np.logical_and(noc[:, :, 0] == 0, noc[:, :, 1] == 0), noc[:, :, 2] == 0))
        return df, np.ascontiguousarray(np.transpose(texture, (2, 0, 1))), np.ascontiguousarray(np.transpose(normal, (2, 0, 1))), np.ascontiguousarray(np.transpose(noc, (2, 0, 1))), mask_texture[np.newaxis, :, :]

    def load_view_dependent_data_from_disk(self, item_index, view_index):
        item = self.items[item_index]
        image_path = self.path_to_dataset / item / f"rgb_{view_index:03d}.png"
        noc_render_path = self.path_to_dataset / item / f"noc_render_{view_index:03d}.png"
        mask_path = self.path_to_dataset / item / f"silhoutte_{view_index:03d}.png"
        partial_texture_path = self.path_to_dataset / item / f"partial_texture_{view_index:03d}.png"
        with Image.open(image_path) as render_im:
            render = self.from_rgb(np.array(render_im)).astype(np.float32)
        with Image.open(noc_render_path) as render_noc:
            noc_render = np.array(render_noc).astype(np.float32)
        with Image.open(partial_texture_path) as partial_tex:
            partial_texture = self.from_rgb(np.array(partial_tex)).astype(np.float32)
        with Image.open(mask_path) as mask_im:
            mask_render = np.array(mask_im)
            mask_render = np.logical_and(np.logical_and(mask_render[:, :, 0] < 50, mask_render[:, :, 1] < 50), mask_render[:, :, 2] < 50)
        render[~mask_render, :] = 0
        noc_render[~mask_render, :] = 0
        render = render[:, :, :3]
        noc_render = noc_render[:, :, :3]
        return np.ascontiguousarray(np.transpose(render, (2, 0, 1))), np.ascontiguousarray(np.transpose(noc_render, (2, 0, 1))), mask_render[np.newaxis, :, :], np.ascontiguousarray(np.transpose(partial_texture, (2, 0, 1)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        item = self.items[index]
        view_index = random.randint(0, self.views_per_shape - 1)
        if self.preload:
            df, texture, normal, noc, mask_texture = self.preload_dict[item]['df'], self.preload_dict[item]['texture'], self.preload_dict[item]['normal'], self.preload_dict[item]['noc'], self.preload_dict[item]['mask_texture']
            render_list, noc_render_list, mask_render_list, partial_texture_list = self.preload_dict[item]['render'], self.preload_dict[item]['noc_render'], self.preload_dict[item]['mask_render'], self.preload_dict[item]['partial_texture']
            render = render_list[view_index]
            noc_render = noc_render_list[view_index]
            mask_render = mask_render_list[view_index]
            partial_texture = partial_texture_list[view_index]
        else:
            df, texture, normal, noc, mask_texture = self.load_view_independent_data_from_disk(index)
            render, noc_render, mask_render, partial_texture = self.load_view_dependent_data_from_disk(index, view_index)
        return {
            'name': f'{item}',
            'view_index': view_index,
            'df': df,
            'texture': texture,
            'normal': normal,
            'noc': noc,
            'noc_render': noc_render,
            'mask_texture': mask_texture.astype(np.float32),
            'render': render,
            'mask_render': mask_render.astype(np.float32),
            'partial_texture': partial_texture
        }

    def apply_batch_transforms(self, batch):
        items_color = ['texture', 'render', 'partial_texture']
        items_non_color = ['normal', 'noc', 'noc_render']
        for item in items_non_color:
            batch[item] = batch[item] / 255 - 0.5
        if self.color_space == 'rgb':
            for item in items_color:
                batch[item] = batch[item] / 255 - 0.5
        elif self.color_space == 'lab':
            for item in items_color:
                batch[item][:, 0, :, :] = batch[item][:, 0, :, :] / 100 - 0.5
                batch[item][:, 1, :, :] = batch[item][:, 1, :, :] / 256
                batch[item][:, 2, :, :] = batch[item][:, 2, :, :] / 256

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
            item = np.clip(self.to_rgb(item), 0, 255) / 255
        return item

    @staticmethod
    def move_batch_to_gpu(batch, device):
        batch['texture'] = batch['texture'].to(device)
        batch['normal'] = batch['normal'].to(device)
        batch['noc'] = batch['noc'].to(device)
        batch['noc_render'] = batch['noc_render'].to(device)
        batch['mask_texture'] = batch['mask_texture'].to(device)
        batch['render'] = batch['render'].to(device)
        batch['mask_render'] = batch['mask_render'].to(device)
        batch['partial_texture'] = batch['partial_texture'].to(device)
        if type(batch['df']) != list:
            batch['df'] = batch['df'].to(device)

    def visualize_sample_pyplot(self, texture, normal, noc, mask_texture, render, noc_render, mask_render, partial_texture):
        import matplotlib.pyplot as plt
        [texture, render, partial_texture], [normal, noc, noc_render], [mask_texture, mask_render] = self.convert_data_for_visualization([texture, render, partial_texture], [normal, noc, noc_render], [mask_texture, mask_render])
        rows, cols = 2, 4
        f, axarr = plt.subplots(2, 4, figsize=(16, 4))
        items = [[texture, normal, render, noc_render], [noc, mask_texture, mask_render, partial_texture]]
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
