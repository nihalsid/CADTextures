from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from PIL import Image


def read_list(path):
    path = Path(path)
    if path.exists():
        return [x.strip() for x in path.read_text().splitlines()]
    print(f'{path} does not exist, returning empty list.')
    return []


def write_list(path, listl):
    Path(path).write_text("\n".join(listl))


def print_model_parameter_count(model):
    from ballpark import business
    count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters in {type(model).__name__}: {business(count, precision=3, prefix=True)}")


def move_batch_to_gpu(batch, device, keys):
    for key in keys:
        batch[key] = batch[key].to(device)


def apply_batch_color_transform_and_normalization(batch, items_color, items_non_color, color_space):
    for item in items_non_color:
        batch[item] = batch[item] / 255 - 0.5
    if color_space == 'rgb':
        for item in items_color:
            batch[item] = batch[item] / 255 - 0.5
    elif color_space == 'lab':
        for item in items_color:
            batch[item][:, 0, :, :] = batch[item][:, 0, :, :] / 100 - 0.5
            batch[item][:, 1, :, :] = batch[item][:, 1, :, :] / 256
            batch[item][:, 2, :, :] = batch[item][:, 2, :, :] / 256


def normalize_tensor_color(item, color_space):
    if color_space == 'rgb':
        item = item / 255 - 0.5
    elif color_space == 'lab':
        item[:, 0, :, :] = item[:, 0, :, :] / 100 - 0.5
        item[:, 1, :, :] = item[:, 1, :, :] / 256
        item[:, 2, :, :] = item[:, 2, :, :] / 256
    return item


def denormalize_and_rgb(arr, color_space, to_rgb_func, only_l):
    if color_space == 'rgb':
        arr = (arr + 0.5) * 255
    elif color_space == 'lab':
        arr[:, :, 0] = np.clip((arr[:, :, 0] + 0.5) * 100, 0, 100)
        if only_l:
            arr[:, :, 1:] = 0
        else:
            arr[:, :, 1] = np.clip(arr[:, :, 1] * 256, -128, 127)
            arr[:, :, 2] = np.clip(arr[:, :, 2] * 256, -128, 127)
    return np.clip(to_rgb_func(arr), 0, 255).astype(np.uint8)


def resize_npy_as_image(npy, size):
    return np.array(Image.fromarray(npy).resize((size, size)))


def to_underscore(s):
    return '__'.join(s.split('/'))


def rename_state_dict(state_dict, key):
    new_state_dict = OrderedDict()
    for k in state_dict:
        if k.startswith(key):
            new_state_dict[".".join(k.split(".")[1:])] = state_dict[k]
    return new_state_dict


def load_net_for_eval(net, ckpt_path, rename_prefix):
    ckpt = torch.load(ckpt_path, map_location='cuda:0')
    net.load_state_dict(rename_state_dict(ckpt['state_dict'], rename_prefix))
    net = net.cuda()
    net.eval()
    return net
