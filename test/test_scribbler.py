import torch

from model.scribbler import ResidualBlock, Scribbler, ImageFusionScribbler, ImageAnd3dFusionScribbler
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
    model = ImageFusionScribbler(3, 3, 3, 24, 16)
    print(model(map_tensor, image_tensor).shape)
    print_model_parameter_count(model)


def test_image_3d_fusion_scribbler():
    map_tensor = torch.randn(8, 3, 384, 384)
    image_tensor = torch.randn(8, 3, 256, 256)
    df_tensor = torch.randn(8, 1, 64, 64, 64)
    model = ImageAnd3dFusionScribbler(3, 3, 3, 24, 16, 12)
    print(model(map_tensor, image_tensor, df_tensor).shape)
    print_model_parameter_count(model)


if __name__ == "__main__":
    test_image_3d_fusion_scribbler()
