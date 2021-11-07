import torch
from torch import nn

from model.interleaved_upsample import InterleavedUpsample
from util.misc import print_model_parameter_count


class InterleavedUpsampleExample(nn.Module):

    def __init__(self):
        super().__init__()
        self.in_channels = 3
        self.out_channels = 1
        self.kernel_size = (3, 3)
        self.weight_00 = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]).float()
        self.weight_01 = torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]).float()
        self.weight_10 = torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 1, 0], [0, 0, 0]]]]).float()
        self.weight_11 = torch.tensor([[[[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0], [0, 0, 0]]]]).float()
        self.bias_00 = torch.tensor([0]).float()
        self.bias_01 = torch.tensor([0]).float()
        self.bias_10 = torch.tensor([0]).float()
        self.bias_11 = torch.tensor([-1]).float()

    def create_interleaved_filter(self):
        return torch.cat([self.weight_00, self.weight_01, self.weight_10, self.weight_11], dim=0)

    def create_interleaved_bias(self):
        return torch.cat([self.bias_00, self.bias_01, self.bias_10, self.bias_11], dim=0)

    def forward(self, x):
        b, c, h, w = x.shape
        fold = torch.nn.Fold(output_size=(2 * h, 2 * w), kernel_size=(2, 2), stride=(2, 2))
        # note that expand doesn't use extra memory
        out = nn.functional.conv2d(x.unsqueeze(2).expand(-1, -1, 4, -1, -1).reshape((b, c * 4, h, w)),
                                   self.create_interleaved_filter(),
                                   self.create_interleaved_bias(),
                                   padding='same',
                                   groups=4)
        out = fold(out.reshape(b, self.out_channels * 4, h * w))
        return out


if __name__ == "__main__":
    model_normal = nn.Conv2d(2, 1, (3, 3), padding=1)
    model_interleaved = InterleavedUpsample(2, 1, (3, 3))
    model_interleaved_example = InterleavedUpsampleExample()
    print_model_parameter_count(model_normal)
    print_model_parameter_count(model_interleaved)
    t_in = torch.rand((4, 2, 8, 8))
    print(model_interleaved(t_in).shape)
    t_in = torch.tensor([[[[0, 3], [6, 9]], [[1, 4], [7, 10]], [[2, 5], [8, 11]]]]).float()
    print(t_in.shape)
    print(model_interleaved_example(t_in))
