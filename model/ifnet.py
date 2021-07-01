import torch.nn as nn
import torch
import torch.nn.functional as F


class TEXR(nn.Module):

    # noinspection PyTypeChecker
    def __init__(self, hidden_dim=256):
        super(TEXR, self).__init__()

        self.conv_in = nn.Conv2d(4, 16, 3, padding=1, padding_mode='replicate')  # out: 256 ->m.p. 128
        self.conv_0 = nn.Conv2d(16, 32, 3, padding=1, padding_mode='replicate')  # out: 128
        self.conv_0_1 = nn.Conv2d(32, 32, 3, padding=1, padding_mode='replicate')  # out: 128 ->m.p. 64
        self.conv_1 = nn.Conv2d(32, 64, 3, padding=1, padding_mode='replicate')  # out: 64
        self.conv_1_1 = nn.Conv2d(64, 64, 3, padding=1, padding_mode='replicate')  # out: 64 -> mp 32
        self.conv_2 = nn.Conv2d(64, 128, 3, padding=1, padding_mode='replicate')  # out: 32
        self.conv_2_1 = nn.Conv2d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 32 -> mp 16
        self.conv_3 = nn.Conv2d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 16
        self.conv_3_1 = nn.Conv2d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 16 -> mp 8
        self.conv_4 = nn.Conv2d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 8
        self.conv_4_1 = nn.Conv2d(128, 128, 3, padding=1, padding_mode='replicate')  # out: 8

        feature_size = (4 + 16 + 32 + 64 + 128 + 128 + 128) * 5 + 2
        self.fc_0 = nn.Conv1d(feature_size, hidden_dim * 2, 1)
        self.fc_1 = nn.Conv1d(hidden_dim * 2, hidden_dim, 1)
        self.fc_2 = nn.Conv1d(hidden_dim, hidden_dim, 1)
        self.fc_out = nn.Conv1d(hidden_dim, 3, 1)
        self.actvn = nn.ReLU()

        self.maxpool = nn.MaxPool2d(2)

        self.conv_in_bn = nn.BatchNorm2d(16)
        self.conv0_1_bn = nn.BatchNorm2d(32)
        self.conv1_1_bn = nn.BatchNorm2d(64)
        self.conv2_1_bn = nn.BatchNorm2d(128)
        self.conv3_1_bn = nn.BatchNorm2d(128)
        self.conv4_1_bn = nn.BatchNorm2d(128)

        displacment = 0.0722
        displacments = [[0, 0]]
        for x in range(2):
            for y in [-1, 1]:
                input = [0, 0]
                input[x] = y * displacment
                displacments.append(input)
        self.register_buffer('displacments', torch.Tensor(displacments))

    def forward(self, p, x):
        # x = x.unsqueeze(1)
        p_features = p.transpose(1, -1)
        p = p.unsqueeze(1)
        p = torch.cat([p + d for d in self.displacments], dim=1)
        feature_0 = F.grid_sample(x, p, padding_mode='border', align_corners=True)
        # print(feature_0[:,:,:,0,0])

        net = self.actvn(self.conv_in(x))
        net = self.conv_in_bn(net)
        feature_1 = F.grid_sample(net, p, padding_mode='border', align_corners=True)
        net = self.maxpool(net)  # out 128

        net = self.actvn(self.conv_0(net))
        net = self.actvn(self.conv_0_1(net))
        net = self.conv0_1_bn(net)
        feature_2 = F.grid_sample(net, p, padding_mode='border', align_corners=True)
        net = self.maxpool(net)  # out 64

        net = self.actvn(self.conv_1(net))
        net = self.actvn(self.conv_1_1(net))
        net = self.conv1_1_bn(net)
        feature_3 = F.grid_sample(net, p, padding_mode='border', align_corners=True)
        net = self.maxpool(net)

        net = self.actvn(self.conv_2(net))
        net = self.actvn(self.conv_2_1(net))
        net = self.conv2_1_bn(net)
        feature_4 = F.grid_sample(net, p, padding_mode='border', align_corners=True)
        net = self.maxpool(net)

        net = self.actvn(self.conv_3(net))
        net = self.actvn(self.conv_3_1(net))
        net = self.conv3_1_bn(net)
        feature_5 = F.grid_sample(net, p, padding_mode='border', align_corners=True)
        net = self.maxpool(net)

        net = self.actvn(self.conv_4(net))
        net = self.actvn(self.conv_4_1(net))
        net = self.conv4_1_bn(net)
        feature_6 = F.grid_sample(net, p, padding_mode='border', align_corners=True)

        # here every channel corresponse to one feature.
        features = torch.cat((feature_0, feature_1, feature_2, feature_3, feature_4, feature_5, feature_6),
                             dim=1)  # (B, features, 1,7,sample_num)
        shape = features.shape
        features = torch.reshape(features, (shape[0], shape[1] * shape[2], shape[3]))  # (B, featues_per_sample, samples_num)
        features = torch.cat((features, p_features), dim=1)  # (B, featue_size, samples_num) samples_num 0->0,...,N->N
        # print('features: ', features[:,:,:3])

        net = self.actvn(self.fc_0(features))
        net = self.actvn(self.fc_1(net))
        net = self.actvn(self.fc_2(net))
        out = self.fc_out(net)
        # out = net.squeeze(1)

        return out
