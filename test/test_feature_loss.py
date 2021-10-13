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

    feature_loss_helper = FeatureLossHelper(['relu4_1'], ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'], 'rgb')
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
                    loss = feature_loss_helper.calculate_feature_loss(content_img, input_img).mean()
                elif loss_type == 'style':
                    loss_content = feature_loss_helper.calculate_feature_loss(content_img, input_img).mean()
                    loss_maps = feature_loss_helper.calculate_style_loss(style_img, input_img)
                    loss = functools.reduce(lambda x, y: x.mean() + y.mean(), loss_maps) * 1e6 + loss_content
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
    test_feature_loss('content')
