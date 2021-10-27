import functools

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from util.feature_loss import FeatureLossHelper


def test_feature_loss(loss_type='content'):
    imsize = 128

    loader = transforms.Compose([
        transforms.Resize(imsize),
        transforms.ToTensor()])

    def image_loader(image_name):
        image = Image.open(image_name)
        image = loader(image).unsqueeze(0) - 0.5
        return image[:, :3, :, :].float().cuda()

    unloader = transforms.ToPILImage()

    plt.ion()

    content_img = image_loader("images/dancing.png")
    style_img = image_loader("images/picasso.jpg")

    def imshow(tensor, title=None):
        image = tensor.cpu().clone() + 0.5  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    if loss_type == 'content':
        input_img = torch.randn(content_img.data.size(), device='cuda:0') - 0.5
    else:
        input_img = content_img.clone()

    def get_input_optimizer(_input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = torch.optim.Adam([_input_img.requires_grad_()], lr=0.01)
        return optimizer

    content_weights = [1 / 8, 1 / 4, 1 / 2, 1]
    style_weights = [1 / 32, 1 / 16, 1 / 8, 1 / 4]
    feature_loss_helper = FeatureLossHelper(['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], 'rgb')
    feature_loss_helper.move_to_device(torch.device('cuda:0'))

    def run_style_transfer(content_img, style_img, input_img, num_steps=300):
        optimizer = get_input_optimizer(input_img)
        print('Optimizing..')
        run = [0]
        while run[0] <= num_steps:
            def closure():
                input_img.data.clamp_(-0.5, 0.5)
                optimizer.zero_grad()
                loss = None
                if loss_type == 'content':
                    loss_maps = feature_loss_helper.calculate_feature_loss(content_img, input_img)
                    weighted_loss_map = loss_maps[0].mean() * content_weights[0]
                    for loss_map_idx in range(1, len(loss_maps)):
                        weighted_loss_map += loss_maps[loss_map_idx].mean() * content_weights[loss_map_idx]
                    loss = weighted_loss_map
                elif loss_type == 'style':
                    loss_maps_c = feature_loss_helper.calculate_feature_loss(content_img, input_img)
                    weighted_loss_map_c = loss_maps_c[0].mean() * content_weights[0]
                    for loss_map_idx in range(1, len(loss_maps_c)):
                        weighted_loss_map_c += loss_maps_c[loss_map_idx].mean() * content_weights[loss_map_idx]
                    loss_maps_s = feature_loss_helper.calculate_style_loss(style_img, input_img)
                    weighted_loss_map_s = loss_maps_s[0].mean() * style_weights[0]
                    for loss_map_idx in range(1, len(loss_maps_s)):
                        weighted_loss_map_s += loss_maps_s[loss_map_idx].mean() * style_weights[loss_map_idx]
                    loss = weighted_loss_map_s * 1e7 + weighted_loss_map_c * 1e-1
                loss.backward()
                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}: {}".format(run, loss.item()))
                return loss
            optimizer.step(closure)
        input_img.data.clamp_(-0.5, 0.5)
        return input_img

    output = run_style_transfer(content_img, style_img, input_img, num_steps=1000)

    plt.figure()
    imshow(output, title='Output Image')
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    test_feature_loss('style')
