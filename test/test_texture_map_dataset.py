from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import hydra

from dataset.texture_map_dataset import TextureMapDataset
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


@hydra.main(config_path='../config', config_name='base')
def main_app(config):
    test_texture_map_dataset(config, True)


if __name__ == '__main__':
    main_app()
