from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import hydra

from dataset.texture_completion_dataset import TextureCompletionDataset
from dataset.texture_map_dataset import TextureMapDataset
from dataset.texture_patch_dataset import TexturePatchDataset
from util.timer import Timer


def test_texture_map_dataset(config, visualize):
    dataset = TextureMapDataset(config, 'train', {})
    print("Length of dataset:", len(dataset))
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    with Timer('Dataloader'):
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            dataset.apply_batch_transforms(batch)
            TextureMapDataset.move_batch_to_gpu(batch, torch.device('cuda'))
    if visualize:
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            dataset.apply_batch_transforms(batch)
            for ii in range(batch['texture'].shape[0]):
                dataset.visualize_sample_pyplot(batch['texture'][ii].numpy(), batch['normal'][ii].numpy(), batch['noc'][ii].numpy(),
                                                batch['mask_texture'][ii].numpy(), batch['render'][ii].numpy(), batch['noc_render'][ii].numpy(), batch['mask_render'][ii].numpy(), batch['partial_texture'][ii].numpy())


def test_texture_completion_dataset(config, visualize):
    dataset = TextureCompletionDataset(config, 'train')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=4, pin_memory=True)
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        dataset.apply_batch_transforms(batch)
        TextureCompletionDataset.move_batch_to_gpu(batch, torch.device('cuda'))
    if visualize:
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            dataset.apply_batch_transforms(batch)
            for ii in range(batch['target'].shape[0]):
                dataset.visualize_sample_pyplot(batch['input'][ii].numpy(), batch['target'][ii].numpy(), batch['mask'][ii].numpy(), batch['missing'][ii].numpy())


def test_texture_patch_dataset(config, visualize):
    dataset = TexturePatchDataset(config, "dtd-cracked_cracked_0004", 0, 25 * 3, 4)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=4, pin_memory=True)
    for batch in tqdm(dataloader):
        dataset.apply_batch_transforms(batch)
        TexturePatchDataset.move_batch_to_gpu(batch, torch.device('cuda'))
    if visualize:
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            dataset.apply_batch_transforms(batch)
            generated = dataset.get_patch_from_tensor(batch['target'])
            for ii in range(batch['target'].shape[0]):
                dataset.visualize_sample_pyplot(batch['input'][ii].numpy(), batch['target'][ii].numpy(), batch['mask_texture'][ii].numpy(), batch['mask_missing'][ii].numpy(), batch['input_patch'][ii].numpy(), generated[ii].numpy())


@hydra.main(config_path='../config', config_name='refinement')
def main_test_texture_map_ds(config):
    test_texture_map_dataset(config, True)


@hydra.main(config_path='../config', config_name='completion')
def main_test_texture_completion_ds(config):
    test_texture_completion_dataset(config, True)


@hydra.main(config_path='../config', config_name='gan_optimize')
def main_test_texture_patch_ds(config):
    test_texture_patch_dataset(config, True)


if __name__ == '__main__':
    main_test_texture_map_ds()
