from model.vqgan import VQGAN
from PIL import Image
import torch
import numpy as np


def test_vqgan():
    import matplotlib.pyplot as plt

    ckpt = "/rhome/ysiddiqui/taming-transformers/logs/2021-07-14T01-39-41_cubetextures_vqgan/checkpoints/last.ckpt"
    model = VQGAN(ckpt).cuda()
    normals = Image.open("images/surface_normals.png").convert("RGB")
    normals = np.array(normals.resize((128, 128), Image.NEAREST))
    mask = np.logical_or(np.logical_or(normals[:, :, 0] != 0, normals[:, :, 1] != 0), normals[:, :, 2] != 0)
    image = Image.open("images/surface_texture_01.png")
    image = image.resize((128, 128), Image.ANTIALIAS)
    image = np.array(image) * mask[:, :, np.newaxis]
    plt.imshow(image)
    plt.show()
    image = image.astype(np.uint8) / 127.5 - 1
    x = (torch.from_numpy(image).permute((2, 0, 1)).unsqueeze(0)).float().cuda()
    x_rec = model(x)
    image = (torch.clamp(x_rec, -1, 1) * 0.5 + 0.5).squeeze(0).permute((1, 2, 0)).detach().cpu().numpy()
    plt.imshow(image)
    plt.show()


if __name__ == "__main__":
    test_vqgan()
