import torch

from model.attention import AttentionFeatureEncoder, AttentionBlock, PatchedAttentionBlock
from model.discriminator import TextureGANDiscriminator
from model.refinement import MainModelInput, MainModelRetrieval, RefinementAttentionTextureGAN
from model.retrieval import Patch16, Patch16Thin, Patch16MLP, FullTexture
from model.texture_gan import ResidualBlock, Scribbler, ImageFusionScribblerSlim, ImageAnd3dFusionScribbler, ScribblerGenerator, ScribblerSlim, TextureGAN, TextureGANSlim
from util.misc import print_model_parameter_count


def test_residual_block():
    input_tensor = torch.randn(8, 256, 16, 16)
    model = ResidualBlock(256, 256, 1, None, (1, 1), True)
    print(model(input_tensor).shape)
    print_model_parameter_count(model)


def test_scribbler():
    input_tensor = torch.randn(8, 5, 384, 384)
    model = Scribbler(5, 3, 24)
    print(model(input_tensor).shape)
    print_model_parameter_count(model)


def test_image_fusion_scribbler():
    map_tensor = torch.randn(8, 3, 384, 384)
    image_tensor = torch.randn(8, 3, 256, 256)
    model = ImageFusionScribblerSlim(3, 3, 3, 12, 12)
    print(model(map_tensor, image_tensor).shape)
    print_model_parameter_count(model)


def test_image_3d_fusion_scribbler():
    map_tensor = torch.randn(8, 3, 384, 384)
    image_tensor = torch.randn(8, 3, 256, 256)
    df_tensor = torch.randn(8, 1, 64, 64, 64)
    model = ImageAnd3dFusionScribbler(3, 3, 3, 24, 16, 12)
    print(model(map_tensor, image_tensor, df_tensor).shape)
    print_model_parameter_count(model)


def test_discriminator():
    texture_tensor = torch.randn(8, 3, 128, 128)
    model = TextureGANDiscriminator(3, 16)
    print(model(texture_tensor).shape)
    print_model_parameter_count(model)


def test_scribbler_generator():
    map_tensor = torch.randn(8, 128)
    model = ScribblerGenerator(128, 3, 16)
    print(model(map_tensor).shape)
    print_model_parameter_count(model)


def test_scribbler_conditional_generator():
    texture_partial = torch.randn(8, 4, 384, 384)
    model = ScribblerSlim(4, 3, 32)
    print(model(texture_partial).shape)
    print_model_parameter_count(model)


def test_texturegan_generator():
    texture_partial = torch.randn(8, 4, 384, 384)
    model = TextureGAN(4, 3, 16)
    print(model(texture_partial).shape)
    print_model_parameter_count(model)


def test_textureganslim_generator():
    texture_partial = torch.randn(8, 4, 384, 384)
    model = TextureGANSlim(4, 3, 16)
    print(model(texture_partial).shape)
    print_model_parameter_count(model)


def test_patch16_retrieval():
    model = Patch16(32, 128)
    t = torch.randn(8, 3, 128, 128)
    print(model(t).shape)
    print_model_parameter_count(model)


def test_attention_fenc():
    model = AttentionFeatureEncoder(32, 96, 8)
    t = torch.randn(8, 32, 8, 8)
    print(model(t).shape)
    print_model_parameter_count(model)


def test_attention_block():
    num_output_channels = 32
    patch_extent = 2
    K = 8
    normalize = True
    use_switching = True
    no_output_mapping = False
    blend = True
    attention = AttentionBlock(num_output_channels, patch_extent, K, normalize, use_switching, no_output_mapping, blend)
    t0 = torch.randn(2, 32, 2, 2)
    t1 = torch.randn(2, K, 32, 2, 2)
    print(attention(t0, t1).shape)
    print(attention.get_features(t0, t0)[0].shape, attention.get_features(t0, t0)[1].shape)
    print_model_parameter_count(attention)


def test_patched_attention_block():
    nf, num_patch_x, patch_extent, K = 32, 16, 2, 8
    attention_block = AttentionBlock(nf, patch_extent, K, True, True, False, True)
    model = PatchedAttentionBlock(nf, num_patch_x, patch_extent, K, attention_block)
    t0 = torch.randn(2, 32, 32, 32)
    t1 = torch.randn(2, K, 32, 32, 32)
    print(model(t0, t1.reshape((2 * K, 32, 32, 32))).shape)
    print_model_parameter_count(model)


def test_main_model_input():
    from torch import nn
    model = MainModelInput(3, 32, nn.BatchNorm2d)
    t = torch.randn(8, 3, 128, 128)
    print(model(t).shape)
    print_model_parameter_count(model)


def test_main_model_retrieval():
    from torch import nn
    model = MainModelRetrieval(3, 32, nn.BatchNorm2d)
    t = torch.randn(8, 3, 8, 8)
    print(model(t).shape)
    print_model_parameter_count(model)


def test_refinement_attention_textureGAN():
    model = RefinementAttentionTextureGAN(3, 3, 32, 8, 8)
    t0 = torch.randn(2, 3, 128, 128)
    t1 = torch.randn(2, 8, 3, 128, 128)
    print(model(t0, t1).shape)
    print_model_parameter_count(model)
    print(model.forward_input(t0).shape)
    print(model.forward_retrievals(t0).shape)
    print(model.forward_features(t0, t0)[0].shape, model.forward_features(t0, t0)[1].shape)


def test_patch16_thin():
    model = Patch16Thin(32, 64)
    t0 = torch.randn(2, 3, 16, 16)
    print(model(t0).shape)
    print_model_parameter_count(model)


def test_patch16_mlp():
    model = Patch16MLP(32, 64)
    t0 = torch.randn(2, 3, 16, 16)
    print(model(t0).shape)
    print_model_parameter_count(model)


def test_full_texture():
    model = FullTexture(32, 64)
    t0 = torch.randn(2, 3, 128, 128)
    print(model(t0).shape)
    print_model_parameter_count(model)


if __name__ == "__main__":
    test_full_texture()
