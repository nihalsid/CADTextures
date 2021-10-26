import random
from functools import wraps
import torch


def clipped(func):
    @wraps(func)
    def wrapped_function(img, *args, **kwargs):
        ret_vals = func(img, *args, **kwargs)
        return torch.clamp(ret_vals[0], -0.5, 0.5), torch.clamp(ret_vals[1], -0.5, 0.5)
    return wrapped_function


@clipped
def rgb_shift(t_in, t_out, r_shift_limit=25, g_shift_limit=25, b_shift_limit=25):
    r_shift = random.uniform(-r_shift_limit, r_shift_limit) / 256
    g_shift = random.uniform(-g_shift_limit, g_shift_limit) / 256
    b_shift = random.uniform(-b_shift_limit, b_shift_limit) / 256
    shifts = [r_shift, g_shift, b_shift]
    for i, shift in enumerate(shifts):
        t_in[:, i] = t_in[:, i] + shift
        t_out[:, i] = t_out[:, i] + shift
    return t_in, t_out


def channel_dropout(t_in, t_out, fill_value=-0.5):
    channel_to_drop = random.randint(0, 2)
    t_in[:, channel_to_drop] = fill_value
    t_out[:, channel_to_drop] = fill_value
    return t_in, t_out


def channel_shuffle(t_in, t_out):
    channels_shuffled = [0, 1, 2]
    random.shuffle(channels_shuffled)
    t_in = t_in[:, channels_shuffled]
    t_out = t_out[:, channels_shuffled]
    return t_in, t_out


@clipped
def random_gamma(t_in, t_out, gamma_limit=(70, 140)):
    gamma = random.randint(gamma_limit[0], gamma_limit[1]) / 100.0
    t_in = torch.pow(t_in + 0.5, gamma) - 0.5
    t_out = torch.pow(t_out + 0.5, gamma) - 0.5
    return t_in, t_out


@clipped
def random_brightness_contrast(t_in, t_out, brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True):
    alpha = 1.0 + random.uniform(-contrast_limit, contrast_limit)
    beta = 0.0 + random.uniform(-brightness_limit, brightness_limit)
    t_in += 0.5
    t_out += 0.5
    t_in *= alpha
    t_out *= alpha
    if beta != 0:
        if brightness_by_max:
            t_in += beta
            t_out += beta
    t_in -= 0.5
    t_out -= 0.5
    return t_in, t_out
