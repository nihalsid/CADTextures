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
        image = loader(image).unsqueeze(0)
        return image[:, :3, :, :].float().cuda()

    unloader = transforms.ToPILImage()

    plt.ion()

    content_img = image_loader("images/dancing.png")
    style_img = image_loader("images/picasso.jpg")

    def imshow(tensor, title=None):
        image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
        image = image.squeeze(0)  # remove the fake batch dimension
        image = unloader(image)
        plt.imshow(image)
        if title is not None:
            plt.title(title)
        plt.pause(0.001)  # pause a bit so that plots are updated

    if loss_type == 'content':
        input_img = torch.randn(content_img.data.size(), device='cuda:0')
    else:
        input_img = content_img.clone()

    def get_input_optimizer(_input_img):
        # this line to show that input is a parameter that requires a gradient
        optimizer = torch.optim.Adam([_input_img.requires_grad_()], lr=0.01)
        return optimizer

    layers = ['relu3_2', 'relu4_2'] if loss_type == 'style' else ['relu4_2']
    feature_loss_helper = FeatureLossHelper(layers)
    feature_loss_helper.move_to_device(torch.device('cuda:0'))

    def run_style_transfer(content_img, style_img, input_img, num_steps=300):
        optimizer = get_input_optimizer(input_img)
        print('Optimizing..')
        weights = torch.ones([1, 1, content_img.shape[2], content_img.shape[3]]).cuda()
        run = [0]
        while run[0] <= num_steps:
            def closure():
                input_img.data.clamp_(0, 1)
                optimizer.zero_grad()
                loss = None
                if loss_type == 'content':
                    loss = feature_loss_helper.calculate_feature_loss(content_img, input_img, weights).mean()
                elif loss_type == 'style':
                    loss_content = feature_loss_helper.calculate_feature_loss(content_img, input_img, weights).mean()
                    loss_maps = feature_loss_helper.calculate_style_loss(style_img, input_img, weights)
                    loss = (loss_maps[0].mean() + loss_maps[1].mean()) * 1e6 + loss_content
                loss.backward()
                run[0] += 1
                if run[0] % 50 == 0:
                    print("run {}: {}".format(run, loss.item()))
                return loss
            optimizer.step(closure)
        input_img.data.clamp_(0, 1)
        input_img_l, _, _ = torch.chunk(input_img, 3, dim=1)
        input_img_lll = torch.cat((input_img_l, input_img_l, input_img_l), 1)
        return input_img_lll

    output = run_style_transfer(content_img, style_img, input_img, num_steps=3000)

    plt.figure()
    imshow(output, title='Output Image')
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    test_feature_loss('style')
