from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from dataset.texture_end2end_dataset import TextureEnd2EndDataset


def test_tiling():

    with Image.open("data/SingleShape/CubeTextures/dtd-cracked_cracked_0056/inv_partial_mask_000.png") as texture_im:
        arr = np.array(texture_im)
        mask_partial = np.logical_or(np.logical_or(arr[:, :, 0] != 0, arr[:, :, 1] != 0), arr[:, :, 2] != 0)

    with Image.open("data/SingleShape/CubeTextures/dtd-cracked_cracked_0056/inv_partial_texture_000.png") as texture_im:
        partial_texture = np.transpose(np.array(texture_im)[:, :, :3], (2, 0, 1))

    partial_texture_completed = TextureEnd2EndDataset.complete_partial_naive(partial_texture, mask_partial)

    f, axarr = plt.subplots(1, 2, figsize=(8, 4))
    axarr[0].imshow(np.transpose(partial_texture.astype(np.uint8), (1, 2, 0)))
    axarr[0].axis('off')
    axarr[1].imshow(np.transpose(partial_texture_completed.astype(np.uint8), (1, 2, 0)))
    axarr[1].axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_tiling()
