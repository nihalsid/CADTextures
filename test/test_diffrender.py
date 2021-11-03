import random
from pathlib import Path
import numpy as np
import torch
import torch_scatter
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

from model.differentiable_renderer import DifferentiableRenderer, intrinsic_to_projection, transform_pos
import trimesh
from PIL import Image

from util.feature_loss import FeatureLossHelper

data_path_0 = Path("data/SingleShape/CubeTexturesForGraph/coloredbrodatz_D48_COLORED/render")
data_path_1 = Path("data/SingleShape/CubeTexturesForGraph/coloredbrodatz_D24_COLORED/render")


def test_diffrender():
    cameras = np.load(data_path_0 / "camera_135_045.npz")
    projection_matrix = intrinsic_to_projection(cameras['cam_intrinsic']).cuda()
    mesh = trimesh.load(data_path_0.parent / "model_normalized.obj", process=False)
    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    indices = torch.from_numpy(mesh.faces).int().cuda()
    colors = torch.from_numpy(mesh.visual.vertex_colors).float().cuda() / 255
    world2cam = np.linalg.inv(cameras['cam_extrinsic'])
    world2cam = torch.from_numpy(world2cam).float().cuda()
    vertices = transform_pos(vertices, projection_matrix, world2cam)
    renderer = DifferentiableRenderer(1024)
    rendered_color = renderer.render(vertices, indices, colors)
    Image.fromarray((rendered_color[0, :, :, :3].cpu().numpy() * 255).astype(np.uint8)).save(f'test_render.png')
    print(rendered_color.shape)


def test_diffrender_optimization(max_iter):
    all_cameras = [np.load(str(x)) for x in data_path_0.iterdir() if x.name.startswith('camera')]
    content_weights = [1 / 8, 1 / 4, 1 / 2, 1]
    style_weights = [1 / 32, 1 / 16, 1 / 8, 1 / 4]
    feature_loss_helper = FeatureLossHelper(['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'], 'rgb')
    feature_loss_helper.move_to_device(torch.device('cuda:0'))
    projection_matrix = intrinsic_to_projection(all_cameras[0]['cam_intrinsic']).cuda()
    mesh = trimesh.load(data_path_0.parent / "model_normalized.obj", process=False)
    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    indices = torch.from_numpy(mesh.faces).int().cuda()
    colors = torch.from_numpy(mesh.visual.vertex_colors).float().cuda() / 255
    # colors_opt = torch.rand_like(colors)[:, :3].contiguous()
    colors_opt = torch.clone(colors)[:, :3].contiguous()
    optimizer = torch.optim.Adam([colors_opt.requires_grad_()], lr=0.0075)
    style_image = torch.from_numpy(np.array(Image.open("test/images/picasso.jpg").resize((128, 128))) / 255).float().cuda().permute((2, 0, 1)).unsqueeze(0)
    renderer = DifferentiableRenderer(128)
    gif_images = []
    for idx in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        world2cam_7 = torch.from_numpy(np.linalg.inv(all_cameras[7]['cam_extrinsic'])).float().cuda()
        world2cam = np.linalg.inv(random.choice(all_cameras)['cam_extrinsic'])
        world2cam = torch.from_numpy(world2cam).float().cuda()
        vertices_opt = transform_pos(vertices, projection_matrix, world2cam)
        vertices_base = transform_pos(vertices, projection_matrix, world2cam_7)
        rendered_color_pred = renderer.render(vertices_opt, indices, colors_opt)
        rendered_color_pred_base = renderer.render(vertices_base, indices, colors_opt).detach()
        rendered_color_gt = renderer.render(vertices_opt, indices, colors)
        visible_mask = rendered_color_gt[:, :, :, 3]
        rendered_color_gt = rendered_color_gt[:, :, :, :3]

        input_image = rendered_color_pred.permute((0, 3, 1, 2)) - 0.5
        loss_maps_c = feature_loss_helper.calculate_feature_loss(rendered_color_gt.permute((0, 3, 1, 2)) - 0.5, input_image)
        weighted_loss_map_c = loss_maps_c[0].mean() * content_weights[0]
        for loss_map_idx in range(1, len(loss_maps_c)):
            weighted_loss_map_c += loss_maps_c[loss_map_idx].mean() * content_weights[loss_map_idx]
        style_image_masked = style_image * visible_mask.unsqueeze(1).expand(-1, 3, -1, -1) - 0.5
        loss_maps_s = feature_loss_helper.calculate_style_loss(style_image_masked, input_image)
        weighted_loss_map_s = loss_maps_s[0].mean() * style_weights[0]
        for loss_map_idx in range(1, len(loss_maps_s)):
            weighted_loss_map_s += loss_maps_s[loss_map_idx].mean() * style_weights[loss_map_idx]
        loss = weighted_loss_map_s * 1e7 + weighted_loss_map_c * 1e-1

        # loss = torch.abs(rendered_color_gt - rendered_color_pred).mean()
        loss.backward()
        optimizer.step()
        print(f'[{idx + 1}/{max_iter}]: {loss.item()}')
        if idx % 10 == 0:
            gif_images.append((torch.cat([rendered_color_gt[0], rendered_color_pred[0], rendered_color_pred_base[0, :, :, :3]], dim=1).detach().cpu().numpy() * 255).astype(np.uint8))
    clip = ImageSequenceClip(gif_images, fps=50)
    clip.write_gif("test_opt.gif", verbose=False, logger=None)
    trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces, vertex_colors=colors_opt.detach().cpu().numpy(), process=False).export("test_opt.obj")


def test_diff_render_batch():
    cameras_0 = np.load(data_path_0 / "camera_135_045.npz")
    cameras_1 = np.load(data_path_0 / "camera_135_135.npz")
    projection_matrix = intrinsic_to_projection(cameras_0['cam_intrinsic']).cuda()
    mesh_0 = trimesh.load(data_path_0.parent / "model_normalized.obj", process=False)
    mesh_1 = trimesh.load(data_path_1.parent / "model_normalized.obj", process=False)
    vertices_0 = torch.from_numpy(mesh_0.vertices).float().cuda()
    indices_0 = torch.from_numpy(mesh_0.faces).int().cuda()
    colors_0 = torch.from_numpy(mesh_0.visual.vertex_colors).float().cuda() / 255
    world2cam_0 = np.linalg.inv(cameras_0['cam_extrinsic'])
    world2cam_0 = torch.from_numpy(world2cam_0).float().cuda()
    vertices_0 = transform_pos(vertices_0, projection_matrix, world2cam_0)
    vertices_1 = torch.from_numpy(mesh_1.vertices).float().cuda()
    indices_1 = torch.from_numpy(mesh_1.faces).int().cuda() + vertices_1.shape[0]
    colors_1 = torch.from_numpy(mesh_1.visual.vertex_colors).float().cuda() / 255
    world2cam_1 = np.linalg.inv(cameras_1['cam_extrinsic'])
    world2cam_1 = torch.from_numpy(world2cam_1).float().cuda()
    vertices_1 = transform_pos(vertices_1, projection_matrix, world2cam_1)
    renderer = DifferentiableRenderer(1024)
    stacked_vertices = torch.cat([vertices_0, vertices_1], dim=0)
    stacked_indices = torch.cat([indices_0, indices_1], dim=0)
    stacked_colors = torch.cat([colors_0, colors_1], dim=0)
    rendered_color = renderer.render(stacked_vertices, stacked_indices, stacked_colors, ranges=torch.tensor([[0, indices_0.shape[0]], [indices_0.shape[0], indices_1.shape[0]]]).int())
    Image.fromarray((rendered_color[0, :, :, :3].cpu().numpy() * 255).astype(np.uint8)).save(f'test_render_0.png')
    Image.fromarray((rendered_color[1, :, :, :3].cpu().numpy() * 255).astype(np.uint8)).save(f'test_render_1.png')
    print(rendered_color.shape)


def test_quad_batched(max_iter):
    data_render = Path("data/SingleShape/CubeTexturesForGraph/coloredbrodatz_D48_COLORED/render")
    data_mesh = Path("data/SingleShape/CubeTexturesForGraphQuad/coloredbrodatz_D48_COLORED/")
    all_cameras = [np.load(str(x)) for x in data_render.iterdir() if x.name.startswith('camera')]
    projection_matrix = intrinsic_to_projection(all_cameras[0]['cam_intrinsic']).cuda()

    cameras = np.load(data_render / "camera_135_135.npz")
    projection_matrix = intrinsic_to_projection(cameras['cam_intrinsic']).cuda()
    mesh = trimesh.load(data_mesh / "model_normalized.obj", process=False)

    vertices = torch.from_numpy(mesh.vertices).float().cuda()
    indices_quad = torch.from_numpy(mesh.faces).int().cuda()
    colors_vertices_original = torch.from_numpy(mesh.visual.vertex_colors).float().cuda()[:, :3] / 255

    # get face colors
    face_colors_opt = torch.rand_like(colors_vertices_original[indices_quad.long(), :].mean(1))
    face_colors_gt = colors_vertices_original[indices_quad.long(), :].mean(1)

    colors_vertices_gt = torch.zeros((vertices.shape[0], 3)).to(face_colors_gt.device)
    torch_scatter.scatter_mean(face_colors_gt.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3), indices_quad.reshape(-1).long(), dim=0, out=colors_vertices_gt)

    # indices triangles
    indices_triangles = torch.cat([indices_quad[:, [0, 1, 2]], indices_quad[:, [0, 2, 3]]], 0)

    renderer = DifferentiableRenderer(128)

    optimizer = torch.optim.Adam([face_colors_opt.requires_grad_()], lr=0.00075)

    gif_images = []
    for idx in range(max_iter):
        optimizer.zero_grad(set_to_none=True)
        
        world2cam = np.linalg.inv(random.choice(all_cameras)['cam_extrinsic'])
        world2cam = torch.from_numpy(world2cam).float().cuda()
        vertices_opt = transform_pos(vertices, projection_matrix, world2cam)

        # face colors to vertex colors
        colors_vertices_pred = torch.zeros_like(colors_vertices_gt)
        torch_scatter.scatter_mean(face_colors_opt.unsqueeze(1).expand(-1, 4, -1).reshape(-1, 3), indices_quad.reshape(-1).long(), dim=0, out=colors_vertices_pred)

        rendered_color_pred = renderer.render(vertices_opt, indices_triangles, colors_vertices_pred)
        rendered_color_gt = renderer.render(vertices_opt, indices_triangles, colors_vertices_gt)

        loss = torch.abs(rendered_color_gt - rendered_color_pred).mean()

        loss.backward()
        optimizer.step()

        print(f'[{idx + 1}/{max_iter}]: {loss.item()}')
        if idx % 100 == 0:
            gif_images.append((torch.cat([rendered_color_gt[0], rendered_color_pred[0]], dim=1).detach().cpu().numpy() * 255).astype(np.uint8))

    clip = ImageSequenceClip(gif_images, fps=50)
    clip.write_gif("test_opt.gif", verbose=False, logger=None)


if __name__ == '__main__':
    test_quad_batched(5000)
