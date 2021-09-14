import torch


class RegressionLossHelper:

    def __init__(self, regression_loss_type):
        self.l2_loss = torch.nn.MSELoss(reduction='none')
        if regression_loss_type == 'l1':
            self.calculate_loss = self.calculate_l1_loss
        elif regression_loss_type == 'l2':
            self.calculate_loss = self.calculate_l2_loss

    # noinspection PyMethodMayBeStatic
    def calculate_l1_loss(self, target, prediction):
        return self.l1_loss(prediction, target)

    def calculate_l2_loss(self, target, prediction):
        return self.l2_loss(prediction, target)
