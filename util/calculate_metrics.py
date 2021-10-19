import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

from util.feature_loss import FeatureLossHelper
from util.regression_loss import RegressionLossHelper

if __name__ == "__main__":

    loss_criterion = RegressionLossHelper('l1')
    l1_criterion = RegressionLossHelper('l1')
    l2_criterion = RegressionLossHelper('l2')
    feature_loss_helper = FeatureLossHelper(['relu4_2'], ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'], 'rgb')
    feature_loss_helper.move_to_device(torch.device('cuda:0'))

    all_items = [x for x in Path("runs/19102246_end2end_fast_dev/visualization/epoch_0015/val_vis").iterdir() if x.name.startswith("pred_")]
    mask = np.array(Image.open("data/SingleShape/CubeTextures/coloredbrodatz_D48_COLORED/surface_normals.png").resize((128, 128), Image.NEAREST))
    mask = torch.from_numpy(np.logical_or(np.logical_or(mask[:, :, 0] != 0, mask[:, :, 1] != 0), mask[:, :, 2] != 0).reshape(-1, 1)).float().cuda()
    loss_total_val_l1 = 0
    loss_total_val_l2 = 0
    content_loss = 0
    style_loss = 0

    for item in tqdm(all_items):
        prediction = torch.from_numpy(np.array(Image.open(item))).cuda().float().reshape((-1, 3)) / 255 - 0.5
        target = torch.from_numpy(np.array(Image.open(item.parent / item.name.replace('pred_', 'tgt_')))).cuda().float().reshape((-1, 3)) / 255 - 0.5
        loss_total_val_l1 += (l1_criterion.calculate_loss(prediction, target).mean(dim=1) * mask).mean().item()
        loss_total_val_l2 += (l2_criterion.calculate_loss(prediction, target).mean(dim=1) * mask).mean().item()
        prediction = prediction.reshape((128, 128, 3)).unsqueeze(0).permute((0, 3, 1, 2))
        target = target.reshape((128, 128, 3)).unsqueeze(0).permute((0, 3, 1, 2))
        with torch.no_grad():
            content_loss += feature_loss_helper.calculate_feature_loss(target, prediction).mean().item()
            style_loss_maps = feature_loss_helper.calculate_style_loss(target, prediction)
            style_loss += np.mean([style_loss_maps[map_idx].mean().item() for map_idx in range(len(style_loss_maps))])

    for loss_name, loss_var in [("loss_val_l1", loss_total_val_l1), ("loss_val_l2", loss_total_val_l2),
                                ("loss_val_style", style_loss), ("loss_val_content", content_loss)]:
        print(f'{loss_name}: {loss_var / len(all_items):.5f}')
