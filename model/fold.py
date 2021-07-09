import torch
from torch import nn


class Fold2D(nn.Module):

    def __init__(self, num_patch_x, patch_extent, nf):
        super().__init__()
        self.nf = nf
        self.num_patch_x = num_patch_x
        self.patch_extent = patch_extent
        self.fold_0 = torch.nn.Fold(output_size=(num_patch_x * patch_extent, num_patch_x * patch_extent), kernel_size=(patch_extent, patch_extent), stride=(patch_extent, patch_extent))

    def forward(self, x):
        fold_in = x.reshape((-1, self.num_patch_x, self.num_patch_x, self.nf, self.patch_extent, self.patch_extent)).permute((0, 3, 4, 5, 1, 2)).reshape((-1, self.nf * self.patch_extent * self.patch_extent, self.num_patch_x * self.num_patch_x))
        fold_out = self.fold_0(fold_in).reshape((-1, self.nf, self.num_patch_x * self.patch_extent, self.num_patch_x * self.patch_extent))
        return fold_out


class Unfold2D(nn.Module):

    def __init__(self, patch_extent, nf):
        super().__init__()
        self.patch_extent = patch_extent
        self.nf = nf

    def forward(self, x):
        unfold_out = x.unfold(2, self.patch_extent, self.patch_extent).unfold(3, self.patch_extent, self.patch_extent)
        return unfold_out.permute((0, 2, 3, 1, 4, 5)).reshape((-1, self.nf, self.patch_extent, self.patch_extent))


class Unfold2DWithContext(nn.Module):

    def __init__(self, patch_extent, patch_context, nf):
        super().__init__()
        self.patch_extent = patch_extent
        self.patch_context = patch_context
        self.nf = nf

    def forward(self, x):
        unfold_out = x.unfold(2, self.patch_context, self.patch_extent).unfold(3, self.patch_context, self.patch_extent)
        return unfold_out.permute((0, 2, 3, 1, 4, 5)).reshape((-1, self.nf, self.patch_context, self.patch_context))
