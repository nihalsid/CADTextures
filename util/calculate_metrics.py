import shutil

import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from cleanfid import fid
from util.feature_loss import FeatureLossHelper
from util.misc import read_list
from util.regression_loss import RegressionLossHelper
import lpips


path = "/cluster_HDD/gondor/ysiddiqui/CADTextures/runs/22101737_graphnn_plain_attention_quad_25_0.10_1200/all_vis_model_44.ckpt"


def calculate_metrics():

    l1_criterion = RegressionLossHelper('l1')
    l2_criterion = RegressionLossHelper('l2')
    loss_fn_alex = lpips.LPIPS(net='alex').cuda()
    feature_loss_helper = FeatureLossHelper(['relu4_2'], ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1', 'relu5_1'], 'rgb')
    feature_loss_helper.move_to_device(torch.device('cuda:0'))
    all_items = sorted([x for x in Path(path).iterdir() if x.name.startswith("pred_")])
    mask = np.array(Image.open("data/SingleShape/CubeTextures/coloredbrodatz_D48_COLORED/surface_normals.png").resize((128, 128), Image.NEAREST))
    mask = torch.from_numpy(np.logical_or(np.logical_or(mask[:, :, 0] != 0, mask[:, :, 1] != 0), mask[:, :, 2] != 0).reshape(-1, 1)).float().cuda()
    loss_total_val_l1 = 0
    loss_total_val_l2 = 0
    content_loss = 0
    style_loss = 0
    lpips_loss = 0

    Path("/tmp/fid/real").mkdir(exist_ok=True, parents=True)
    Path("/tmp/fid/fake").mkdir(exist_ok=True, parents=True)
    for idx, item in enumerate(tqdm(all_items)):
        shutil.copyfile(item, Path(f"/tmp/fid/fake/{item.name}"))
        shutil.copyfile(item.parent / item.name.replace('pred_', 'tgt_'), Path(f"/tmp/fid/real/{item.name.replace('pred_', 'tgt_')}"))
        prediction = torch.from_numpy(np.array(Image.open(item))).cuda().float().reshape((-1, 3)) / 255 - 0.5
        target = torch.from_numpy(np.array(Image.open(item.parent / item.name.replace('pred_', 'tgt_')))).cuda().float().reshape((-1, 3)) / 255 - 0.5
        prediction = prediction * mask
        target = target * mask
        loss_total_val_l1 += (l1_criterion.calculate_loss(prediction, target)).mean().item()
        loss_total_val_l2 += (l2_criterion.calculate_loss(prediction, target)).mean().item()
        prediction = prediction.reshape((128, 128, 3)).unsqueeze(0).permute((0, 3, 1, 2))
        target = target.reshape((128, 128, 3)).unsqueeze(0).permute((0, 3, 1, 2))
        with torch.no_grad():
            lpips_loss += loss_fn_alex(target * 2, prediction * 2).cpu().item()
            content_loss += feature_loss_helper.calculate_feature_loss(target, prediction).mean().item()
            style_loss_maps = feature_loss_helper.calculate_style_loss(target, prediction)
            style_loss += np.mean([style_loss_maps[map_idx].mean().item() for map_idx in range(len(style_loss_maps))])
    fid_score = fid.compute_fid("/tmp/fid/real", "/tmp/fid/fake", mode="clean")
    shutil.rmtree("/tmp/fid")
    for loss_name, loss_var in [("l1", loss_total_val_l1), ("l2", loss_total_val_l2), ("content", content_loss), ("style", style_loss * 1e3), ("lpips_loss", lpips_loss)]:
        print(f'{loss_name}: {loss_var / len(all_items):.5f}')
    print(f"fid: {fid_score:.5f}")


def visualize_results():
    mask = np.array(Image.open("data/SingleShape/CubeTextures/coloredbrodatz_D48_COLORED/surface_normals.png").resize((128, 128), Image.NEAREST))
    mask = np.logical_or(np.logical_or(mask[:, :, 0] != 0, mask[:, :, 1] != 0), mask[:, :, 2] != 0).reshape((128, 128, 1))
    items = sorted(read_list('data/splits/SingleShape/CubeTextures/official/val_vis.txt'))
    export_ctr = 0
    image_list = []
    plane_suffix = "_000_000"
    cube_suffix = "_135_180"
    prefix = Path(path).parent.name
    for item in items:
        image = (np.array(Image.open(Path(path, f"pred_{item}{plane_suffix}.png"))) * mask).astype(np.uint8)
        image_list.append(image)
        if len(image_list) == 16:
            Image.fromarray(np.concatenate(image_list, axis=1)).save(f"/home/nihalsid/{prefix}_{export_ctr:03d}.jpg")
            export_ctr += 1
            image_list = []
    Image.fromarray(np.concatenate(image_list, axis=1)).save(f"/home/nihalsid/{prefix}_{export_ctr:03d}.jpg")


if __name__ == "__main__":
    calculate_metrics()
    visualize_results()
