from torchvision.models import vgg19


class FeatureLossEvaluator:

    def __init__(self):
        self.model_vgg19 = vgg19(pretrained=True)
        self.model_vgg19.eval()
