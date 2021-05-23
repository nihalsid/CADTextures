from torch import nn
import torch
from model.texture_gan import ResidualBlock, UpsamplingBlock
from model.attention import PatchedAttentionBlock, AttentionBlock, Unfold2D, Fold2D


class MainModelInput(nn.Module):

    def __init__(self, input_nc, ngf, norm):
        super(MainModelInput, self).__init__()

        # noinspection PyTypeChecker
        self.module_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_nc, ngf, 3, 1, 1),
                norm(ngf),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf, ngf),
                nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
                norm(ngf * 2),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf * 2, ngf * 2),
                nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
                norm(ngf * 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf * 4, ngf * 4),
                nn.Conv2d(ngf * 4, ngf * 8, 3, 2, 1),
                norm(ngf * 8),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf * 8, ngf * 8),
                ResidualBlock(ngf * 8, ngf * 8),
                ResidualBlock(ngf * 8, ngf * 8),
                ResidualBlock(ngf * 8, ngf * 8),
                ResidualBlock(ngf * 8, ngf * 8),
            ),
            nn.Sequential(
                UpsamplingBlock(ngf * 8, ngf * 4, 3, 1, 1),
                norm(ngf * 4),
                nn.ReLU(True),
                ResidualBlock(ngf * 4, ngf * 4),
                ResidualBlock(ngf * 4, ngf * 4),
            ),
            nn.Sequential(
                UpsamplingBlock(ngf * 4, ngf * 2, 3, 1, 1),
                norm(ngf * 2),
                nn.ReLU(True),
                ResidualBlock(ngf * 2, ngf * 2),
                ResidualBlock(ngf * 2, ngf * 2),
            ),
            nn.Sequential(
                UpsamplingBlock(ngf * 2, ngf - input_nc, 3, 1, 1),
                nn.ReLU(True),
                ResidualBlock(ngf - input_nc, ngf - input_nc, bn=lambda c: nn.Identity()),
            ),
        ])

    def forward(self, x_in, *_condition):
        x = x_in
        for module in self.module_list:
            x = module(x)
        return torch.cat([x, x_in], 1)


class MainModelRetrieval(nn.Module):

    def __init__(self, input_nc, ngf, norm):
        super(MainModelRetrieval, self).__init__()

        # noinspection PyTypeChecker
        self.module_list = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(input_nc, ngf, 3, 1, 1),
                norm(ngf),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf, ngf),
                nn.Conv2d(ngf, ngf * 2, 3, 2, 1),
                norm(ngf * 2),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf * 2, ngf * 2),
                nn.Conv2d(ngf * 2, ngf * 4, 3, 2, 1),
                norm(ngf * 4),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf * 4, ngf * 4),
                nn.Conv2d(ngf * 4, ngf * 8, 3, 2, 1),
                norm(ngf * 8),
                nn.ReLU(True),
            ),
            nn.Sequential(
                ResidualBlock(ngf * 8, ngf * 8),
                ResidualBlock(ngf * 8, ngf * 8),
            ),
            nn.Sequential(
                UpsamplingBlock(ngf * 8, ngf * 4, 3, 1, 1),
                norm(ngf * 4),
                nn.ReLU(True),
                ResidualBlock(ngf * 4, ngf * 4),
            ),
            nn.Sequential(
                UpsamplingBlock(ngf * 4, ngf * 2, 3, 1, 1),
                norm(ngf * 2),
                nn.ReLU(True),
                ResidualBlock(ngf * 2, ngf * 2),
            ),
            nn.Sequential(
                UpsamplingBlock(ngf * 2, ngf, 3, 1, 1),
                norm(ngf),
                nn.ReLU(True),
                ResidualBlock(ngf, ngf),
            ),
        ])

    def forward(self, x_in):
        x = x_in
        for module in self.module_list:
            x = module(x)
        return x


class RefinementAttentionTextureGAN(nn.Module):

    def __init__(self, input_nc, output_nc, ngf, K, num_patch_x):

        super(RefinementAttentionTextureGAN, self).__init__()
        norm = lambda c: nn.GroupNorm(4, c)
        self.num_patch_x = num_patch_x
        self.unfold_shape = Unfold2D(128 // self.num_patch_x, 3)
        self.fold_features = Fold2D(self.num_patch_x, 128 // self.num_patch_x, ngf)
        self.fold_shape = Fold2D(self.num_patch_x, 128 // self.num_patch_x, 3)
        self.input_feature_extractor = MainModelInput(input_nc, ngf, norm)
        self.retrieval_feature_extractor = MainModelRetrieval(3, ngf, norm)
        attention_block = AttentionBlock(ngf, 8, K, True, True, True, True)
        self.patch_attention = PatchedAttentionBlock(ngf, num_patch_x, 128 // self.num_patch_x, K, attention_block)

        # noinspection PyTypeChecker
        self.decoder = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, 3, 1, 1),
            ResidualBlock(ngf * 2, ngf * 2),
            ResidualBlock(ngf * 2, ngf * 2),
            nn.Conv2d(ngf * 2, output_nc, 3, 1, 1),
        )
        self.tanh = nn.Tanh()

    def forward(self, x, *condition):
        input_features = self.get_input_features(x)
        decoded_features = self.decoder(input_features)
        input_prediction = self.tanh(decoded_features) * 0.5
        retrieval_features = self.get_retrieval_features(condition[0])
        return self.blend_and_decode(input_features, retrieval_features, input_prediction.detach(), condition[0])

    def forward_debug(self, x, *condition):
        input_features = self.get_input_features(x)
        decoded_features = self.decoder(input_features)
        input_prediction = self.tanh(decoded_features) * 0.5
        retrieval_features = self.get_retrieval_features(condition[0])
        return self.blend_and_decode(input_features, retrieval_features, input_prediction.detach(), condition[0], True)

    def forward_features(self, x, target):
        input_features = self.get_input_features(x)
        decoded_features = self.decoder(input_features)
        input_prediction = self.tanh(decoded_features) * 0.5
        return self.patch_attention.get_features(input_prediction.detach(), target)

    def forward_retrievals(self, retrievals):
        retrieval_features = self.get_retrieval_features(retrievals)
        decoded_features = self.decoder(retrieval_features)
        folded_decode = self.fold_shape(decoded_features)
        return self.tanh(folded_decode) * 0.5

    def forward_input(self, x):
        input_features = self.get_input_features(x)
        decoded_features = self.decoder(input_features)
        return self.tanh(decoded_features) * 0.5

    def retrieval_patches(self, retrievals):
        # retrievals are BxKx3x128x128
        s = retrievals.shape[-1]
        # B.Kx3x128x128
        retrievals = retrievals.reshape((-1, 3, s, s))
        # B.K.NPX.NPXx3xPExPE
        retrieval_patches = self.unfold_shape(retrievals)
        return retrieval_patches

    def get_input_features(self, x):
        return self.input_feature_extractor(x)

    def get_retrieval_features(self, retrievals):
        # B.K.NPX.NPXx3xPExPE
        retrieval_patches = self.retrieval_patches(retrievals)
        # B.K.NPX.NPXxNGFxPExPE
        return self.retrieval_feature_extractor(retrieval_patches)

    def blend_and_decode(self, input_features, retrieval_features, inputs, retrievals, debug=False):
        # B.KxNGFxSxS
        retrieval_features = self.fold_features(retrieval_features)
        if debug:
            blended_features, weights, scores, switch = self.patch_attention(input_features, retrieval_features, inputs, retrievals, True)
            x = self.decoder(blended_features)
            return self.tanh(x) * 0.5, weights * switch.unsqueeze(1).expand(-1, weights.shape[1], -1, -1), scores
        else:
            blended_features = self.patch_attention(input_features, retrieval_features, inputs, retrievals)
            x = self.decoder(blended_features)
            return self.tanh(x) * 0.5


def get_model(config):
    return RefinementAttentionTextureGAN(4, 3, config.model.input_texture_ngf, config.dictionary.K, 128 // config.dictionary.patch_size)
