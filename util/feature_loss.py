from torchvision.models import vgg19
import torch


class FeatureLossHelper:

    def __init__(self, layers):
        model_vgg19 = vgg19(pretrained=True)
        model_vgg19.eval()
        self.feature_extractor = FeatureExtractor(model_vgg19.features, layers)
        self.criterion = torch.nn.MSELoss(reduction='none')

    def move_to_device(self, device):
        self.feature_extractor.to(device)

    @staticmethod
    def normalize_for_vgg(x_lll):
        # x_lll is assumed to be normalized between [0, 1]
        x_lll[:, 0, :, :] = (x_lll[:, 0, :, :] - 0.485) / 0.229
        x_lll[:, 1, :, :] = (x_lll[:, 1, :, :] - 0.456) / 0.224
        x_lll[:, 2, :, :] = (x_lll[:, 2, :, :] - 0.406) / 0.225
        return x_lll

    def prepare_vgg_input(self, target, prediction, weights):
        expanded_weights = weights.expand(-1, target.shape[1], -1, -1)
        target_l, _, _ = torch.chunk(target, 3, dim=1)
        prediction_l, _, _ = torch.chunk(prediction, 3, dim=1)
        # noinspection PyTypeChecker
        target_lll = self.normalize_for_vgg((torch.cat((target_l, target_l, target_l), 1) + 0.5) * expanded_weights)
        # noinspection PyTypeChecker
        prediction_lll = self.normalize_for_vgg((torch.cat((prediction_l, prediction_l, prediction_l), 1) + 0.5) * expanded_weights)
        return target_lll, prediction_lll

    def calculate_feature_loss(self, target, prediction, weights):
        target_lll, prediction_lll = self.prepare_vgg_input(target, prediction, weights)
        return self.criterion(self.feature_extractor(prediction_lll)[0], self.feature_extractor(target_lll)[0].detach())

    def calculate_style_loss(self, target, prediction, weights):
        target_lll, prediction_lll = self.prepare_vgg_input(target, prediction, weights)
        features_prediction = self.feature_extractor(prediction_lll)
        features_target = self.feature_extractor(target_lll)
        gram = GramMatrix()
        error_maps = []
        for m in range(len(features_target)):
            gram_y = gram(features_prediction[m])
            gram_s = gram(features_target[m])
            error_maps.append(self.criterion(gram_y, gram_s.detach()))
        return error_maps


class GramMatrix(torch.nn.Module):

    # noinspection PyMethodMayBeStatic
    def forward(self, x):
        B, C, H, W = x.size()
        features = x.view(B, C, H * W)
        G = torch.bmm(features, features.transpose(1, 2))  # compute the gram product
        # normalize the values of the gram matrix
        # by dividing by the number of element in each feature maps.
        return G.div(C * H * W)


class FeatureExtractor(torch.nn.Module):
    # Extract features from intermediate layers of a network

    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.extracted_layers = extracted_layers
        self.submodule = self.create_named_vgg(submodule, extracted_layers[-1])

    @staticmethod
    def create_named_vgg(submodule, break_at):
        model = torch.nn.Sequential()
        i, j = 1, 1
        for layer in submodule.children():
            if isinstance(layer, torch.nn.Conv2d):
                name = 'conv{}_{}'.format(i, j)
            elif isinstance(layer, torch.nn.ReLU):
                name = 'relu{}_{}'.format(i, j)
                layer = torch.nn.ReLU(inplace=False)
                j += 1
            elif isinstance(layer, torch.nn.MaxPool2d):
                name = 'pool_{}'.format(i)
                i += 1
                j = 1
            elif isinstance(layer, torch.nn.BatchNorm2d):
                name = 'bn{}_{}'.format(i, j)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))
            model.add_module(name, layer)
            if name == break_at:
                break
        return model

    def forward(self, x):
        outputs = []
        for name, module in self.submodule.named_children():
            x = module(x)
            if name in self.extracted_layers:
                outputs += [x]
        return outputs
