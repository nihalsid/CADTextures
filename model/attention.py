import torch
from torch import nn

from model.retrieval import Patch16Thin, Patch16MLP


class Conv2dAttentionOutput(nn.Conv2d):

    def __init__(self, nf_in, nf_out):
        # noinspection PyTypeChecker
        super().__init__(nf_in, nf_out, kernel_size=1, stride=1, padding=0)

    def reset_parameters(self) -> None:
        nn.init.dirac_(self.weight)
        with torch.no_grad():
            self.weight[:] = self.weight[:] + torch.randn_like(self.weight[:]) * 0.01
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class Conv2dAttentionFeature(nn.Conv2d):

    def __init__(self, nf_in, nf_out):
        # noinspection PyTypeChecker
        super().__init__(nf_in, nf_out, kernel_size=1, stride=1, padding=0)

    def reset_parameters(self) -> None:
        nn.init.normal_(self.weight, 0, 0.01)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


class AttentionFeatureEncoder(nn.Module):

    def __init__(self, n_in, n_out, e):
        super().__init__()
        self.n_in = n_in * (e ** 2)
        self.n_out = n_out
        print("Attention Feature: ", self.n_in, "-->", self.n_out)
        self.encoder = torch.nn.Sequential(nn.Linear(self.n_in, 512),
                                           nn.LeakyReLU(),
                                           nn.Linear(512, 256),
                                           nn.LeakyReLU(),
                                           nn.Linear(256, 256),
                                           nn.LeakyReLU(),
                                           nn.Linear(256, 128),
                                           nn.LeakyReLU(),
                                           nn.Linear(128, 128),
                                           nn.LeakyReLU(),
                                           nn.Linear(128, self.n_out),)

    def forward(self, x):
        b = x.shape[0]
        return self.encoder(x.reshape((b, self.n_in)))


class AttentionBlock(nn.Module):

    def __init__(self, num_output_channels, patch_extent, K, normalize, use_switching, no_output_mapping, blend):
        super().__init__()
        self.cf_op = num_output_channels
        self.cf_feat = 64
        self.theta = Patch16MLP(num_output_channels, self.cf_feat)
        self.phi = Patch16MLP(num_output_channels, self.cf_feat)
        # self.theta = AttentionFeatureEncoder(num_output_channels, self.cf_feat, patch_extent)
        # self.phi = AttentionFeatureEncoder(num_output_channels, self.cf_feat, patch_extent)
        self.g = Conv2dAttentionOutput(num_output_channels, self.cf_op) if not no_output_mapping else nn.Identity()
        self.o = Conv2dAttentionOutput(self.cf_op, num_output_channels) if not no_output_mapping else nn.Identity()
        self.max = nn.MaxPool1d(kernel_size=K)
        self.init_scale = 40
        self.init_shift = -20
        self.sig_scale = nn.Parameter(torch.ones(1) * self.init_scale)
        self.sig_shift = nn.Parameter(torch.ones(1) * self.init_shift)
        self.sigmoid = nn.Sigmoid()
        self.softmax = torch.nn.Softmax(dim=1)
        self.blend_mode = blend
        self.use_switching = use_switching
        self.normalize = normalize

    def get_features(self, x_im, p_im):
        # x: BxC0xExE, p: BxC0xExE
        b = x_im.shape[0]
        x_feat = self.theta(x_im)
        p_feat = self.phi(p_im)
        x_feat_flat = x_feat.reshape((b, -1))
        p_feat_flat = p_feat.reshape((b, -1))
        if self.normalize:
            x_feat_flat = nn.functional.normalize(x_feat_flat, dim=1)
            p_feat_flat = nn.functional.normalize(p_feat_flat, dim=1)
        return x_feat_flat, p_feat_flat

    def forward(self, x, p, x_im, p_im, return_debug_outputs=False):
        # x: BxC0xExExE, p: BxKxC0xExExE
        b, k, c, e = [p.shape[0], p.shape[1], p.shape[2], p.shape[3]]
        x_feat = self.theta(x_im)
        x_feat_flat = x_feat.reshape((b, -1))
        p_feat = self.phi(p_im.reshape(b * k, -1, e, e))
        p_feat_flat = p_feat.reshape((b, k, -1))
        if self.normalize:
            x_feat_flat = nn.functional.normalize(x_feat_flat, dim=1)
            p_feat_flat = nn.functional.normalize(p_feat_flat, dim=2)
        g_feat_flat = self.g(p.reshape(b * k, -1, e, e)).reshape((b, k, -1, e, e)).reshape((b, k, -1))

        scores = torch.einsum('ij,ijk->ik', x_feat_flat, p_feat_flat.permute((0, 2, 1)))
        print(scores.min(), scores.max())
        switch = self.sigmoid(self.max(scores.view(b, 1, k)).view(b, 1) * self.sig_scale + self.sig_shift) if self.use_switching else 1
        sharpness = (self.cf_feat * e * e * e) * 4
        weights = self.softmax(sharpness * scores)
        # weights = nn.functional.gumbel_softmax(scores, tau=0.00001, hard=True)
        # print(self.max(scores.view(b, 1, k)).view(b, 1)[torch.std(weights, dim=1) >= torch.std(weights, dim=1).max()/2][:5])
        # print(switch[torch.std(weights, dim=1) >= torch.std(weights, dim=1).max()/2][:5])
        # print(weights[torch.std(weights, dim=1) >= torch.std(weights, dim=1).max()/2, :][:5, :])
        weighted_sum = torch.einsum('ij,ijk->ik', weights, g_feat_flat).reshape((b, -1, e, e))  # BxCFxExExE
        patch_attention = self.o(weighted_sum)
        if self.blend_mode:
            output = (x.view(b, c * e * e) * (1 - switch)).view(b, c, e, e) + (patch_attention.view(b, c * e * e) * switch).view(b, c, e, e)
        else:
            output = x + (patch_attention.view(b, c * e * e) * switch).view(b, c, e, e)
        if return_debug_outputs:
            return output, scores.detach(), weights.detach(), switch.detach()
        return output


class PatchedAttentionBlock(nn.Module):

    def __init__(self, nf, num_patch_x, patch_extent, num_nearest_neighbors, attention_block):
        super().__init__()
        self.num_patch_x = num_patch_x
        self.patch_extent = patch_extent
        self.num_nearest_neighbors = num_nearest_neighbors
        self.nf = nf
        self.attention_blocks_layer = attention_block
        self.fold_2d = Fold2D(num_patch_x, patch_extent, self.nf)
        self.unfold_2d = Unfold2D(patch_extent, self.nf)
        self.unfold_2d_im = Unfold2D(patch_extent, 3)

    def get_features(self, x_predicted, x_target):
        x_predicted_feat_ = self.unfold_2d_im(x_predicted)
        x_target_feat_ = self.unfold_2d_im(x_target)
        x_feat_flat, p_feat_flat = self.attention_blocks_layer.get_features(x_predicted_feat_, x_target_feat_)
        return x_feat_flat, p_feat_flat

    def forward(self, x_predicted, x_retrieved, input_im, retrieved_im, debug=False):
        # x_predicted: BxFxSxS
        # x_retrieved: B.KxFxSxS
        shape_dim = x_retrieved.shape[-1]
        x_patch = x_retrieved.reshape(-1, self.nf, shape_dim, shape_dim)
        # x_predicted_feat_: (B.R.RxFxE)
        x_predicted_feat_ = self.unfold_2d(x_predicted)
        # x_patch_feat_: (B.K.R.RxFxExE)
        x_patch_feat_ = self.unfold_2d(x_patch)
        # x_predicted_feat_: (B.R.R.RxKxFxExE)
        x_patch_feat_ = x_patch_feat_.reshape((-1, self.num_nearest_neighbors, self.num_patch_x, self.num_patch_x, self.nf, self.patch_extent, self.patch_extent)).permute(
            (0, 2, 3, 1, 4, 5, 6)).reshape((-1, self.num_nearest_neighbors, self.nf, self.patch_extent, self.patch_extent))

        x_patch_im = retrieved_im.reshape(-1, 3, shape_dim, shape_dim)
        # x_predicted_feat_: (B.R.RxFxE)
        x_predicted_feat_im_ = self.unfold_2d_im(input_im)
        # x_patch_feat_: (B.K.R.RxFxExE)
        x_patch_feat_im_ = self.unfold_2d_im(x_patch_im)
        # x_predicted_feat_: (B.R.R.RxKxFxExE)
        x_patch_feat_im_ = x_patch_feat_im_.reshape((-1, self.num_nearest_neighbors, self.num_patch_x, self.num_patch_x, 3, self.patch_extent, self.patch_extent)).permute(
            (0, 2, 3, 1, 4, 5, 6)).reshape((-1, self.num_nearest_neighbors, 3, self.patch_extent, self.patch_extent))

        # attention_processed: (B.R.RxFxExE)
        weights = scores = switch = None
        if not debug:
            attention_processed = self.attention_blocks_layer(x_predicted_feat_, x_patch_feat_, x_predicted_feat_im_, x_patch_feat_im_)
        else:
            attention_processed, scores, weights, switch = self.attention_blocks_layer(x_predicted_feat_, x_patch_feat_, x_predicted_feat_im_, x_patch_feat_im_, debug)
            weights = weights.reshape((-1, self.num_patch_x, self.num_patch_x, self.num_nearest_neighbors)).permute((0, 3, 1, 2))
            scores = scores.reshape((-1, self.num_patch_x, self.num_patch_x, self.num_nearest_neighbors)).permute((0, 3, 1, 2))
            switch = switch.reshape((-1, self.num_patch_x, self.num_patch_x))
        output_feats = self.fold_2d(attention_processed)
        if debug:
            return output_feats, weights, scores, switch
        return output_feats


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
        return x.unfold(2, self.patch_extent, self.patch_extent).unfold(3, self.patch_extent, self.patch_extent).permute((0, 2, 3, 1, 4, 5)).reshape((-1, self.nf, self.patch_extent, self.patch_extent))
