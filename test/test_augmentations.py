import hydra
import numpy as np
from PIL import Image
from model.augmentation import *
from dataset.graph_mesh_dataset import FaceGraphMeshDataset


@hydra.main(config_path='../config', config_name='graph_nn_test')
def test_augmentations(config):
    dataset = FaceGraphMeshDataset(config, 'val_vis', use_single_view=True, load_to_memory=False)
    for j in range(4):
        for i in range(len(dataset)):
            sample = dataset[i]
            sample.x[:, 3:6], sample.y = random_brightness_contrast(sample.x[:, 3:6], sample.y)
            t_in = ((dataset.to_image(sample.x[:, 3:6]).squeeze(0).permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)
            t_out = ((dataset.to_image(sample.y).squeeze(0).permute((1, 2, 0)).cpu().numpy() + 0.5) * 255).astype(np.uint8)
            Image.fromarray(t_in).save(f'aug_{j:02}_{i:02}_in.png')
            Image.fromarray(t_out).save(f'aug_{j:02}_{i:02}_tgt.png')


if __name__ == "__main__":
    test_augmentations()
