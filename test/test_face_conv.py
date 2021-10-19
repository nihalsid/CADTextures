import torch
from model.graphnet import BigFaceEncoderDecoder, FaceConv


pt_data = torch.load("data/SingleShape/CubeTexturePlaneQuad_FC_processed/coloredbrodatz_D48_COLORED_000_000.pt")


def test_face_conv():
    x = torch.cat([pt_data['input_colors'], pt_data['valid_input_colors'].unsqueeze(-1)], 1)
    face_neighborhood = pt_data['conv_data'][0][0]
    pad_size = pt_data['conv_data'][0][2].shape[0]
    fc = FaceConv(4, 64, 8)
    out = fc(x, face_neighborhood, pad_size)
    print(out.shape)


def test_big_encoder_decoder():
    from util.misc import print_model_parameter_count
    model = BigFaceEncoderDecoder(4, 3, 128, 8)
    print(model)
    print_model_parameter_count(model)
    x = torch.cat([pt_data['input_colors'], pt_data['valid_input_colors'].unsqueeze(-1)], 1)
    face_neighborhoods = [pt_data['conv_data'][i][0] for i in range(len(pt_data['conv_data']))]
    node_counts = [pt_data['conv_data'][i][0].shape[0] for i in range(1, len(pt_data['conv_data']))]
    pad_sizes = [pt_data['conv_data'][i][2].shape[0] for i in range(len(pt_data['conv_data']))]
    pool_maps = pt_data['pool_locations']
    out = model(x, face_neighborhoods[0], node_counts, pool_maps, pad_sizes, face_neighborhoods[1:])
    print(out.shape)


if __name__ == '__main__':
    test_big_encoder_decoder()
