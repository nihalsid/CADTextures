import torch
from model.graphnet import BigFaceEncoderDecoder, FaceConv, SpatialAttentionConv

pt_data = torch.load("data/SingleShape/CubeTexturesForGraphQuad_FC_processed/coloredbrodatz_D48_COLORED_180_270.pt")


def test_face_conv():
    x = torch.zeros((pt_data['conv_data'][4][4].sum(), 1)).float()
    x[:, 0] = torch.FloatTensor(list(range(1, x.shape[0] + 1))).float()
    face_neighborhood = pt_data['conv_data'][4][0]
    is_pad = pt_data['conv_data'][4][4]
    pad_size = pt_data['conv_data'][4][2].shape[0]
    fc = SpatialAttentionConv(1, 64)
    out = fc(x, face_neighborhood, is_pad, pad_size)
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
    test_face_conv()
