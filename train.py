#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import cv2
import numpy as np
import os
import torch
import sys
from datetime import datetime
from argparse import ArgumentParser, Namespace
from random import randint, sample
from tqdm import tqdm
from torch.nn import functional as F
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from arguments import ModelParams, PipelineParams, OptimizationParams
from gaussian_renderer import render, network_gui
from scene import Scene, GaussianModel, Camera
from utils.general_utils import safe_state
from utils.render_utils import tensor2cv, apply_depth_colormap
from utils.image_utils import psnr, render_net_image
from utils.loss_utils import l1_loss, ssim
from utils.graphics_utils import patch_warp, lncc, patch_offsets


def loss_in_neighbor_view(
        view_cur: Camera, view_neighbor: Camera, 
        pts_cur: dict, pts_neigh: dict,
        patch_template: torch.Tensor,
        pixels: torch.Tensor, pixel_noise_threshold: float,
        debug: bool = False
    ) -> dict:

    pts_neighbor_view = pts_cur['pts_world'] @ view_neighbor.world_view_transform[:3, :3] + view_neighbor.world_view_transform[3, :3]
    proj_uvw = pts_neighbor_view @ view_neighbor.intrins
    proj_depth = proj_uvw[:, 2:3]
    proj_uvw = proj_uvw / proj_depth
    proj_uv = torch.stack([
        proj_uvw[:, 0] / (view_neighbor.image_width-1) * 2 - 1,
        proj_uvw[:, 1] / (view_neighbor.image_height-1) * 2 - 1
    ], dim=-1)
    proj_uv_mask = (proj_uv[:, 0] > -1) & (proj_uv[:, 0] < 1) & (proj_uv[:, 1] > -1) & (proj_uv[:, 1] < 1) & (proj_depth.squeeze(1) > 0) 

    # viz proj_depth
    # if debug:
    #     proj_depth_viz = proj_depth / 4
    #     proj_depth_viz = proj_depth_viz.reshape(view_cur.image_height, view_cur.image_width)
    #     proj_depth_viz = cv2.applyColorMap((proj_depth_viz * 255).detach().cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
    #     cv2.imwrite('debug/proj_depth.png', proj_depth_viz)

    proj_uv = proj_uv.reshape(1, -1, 1, 2)

    sampled_depth = F.grid_sample(
        pts_neigh['surf_depth'].unsqueeze(0), proj_uv,
        mode='bilinear', padding_mode='border', align_corners=True,
    ).squeeze()

    # if debug:
    #     depth_diff_viz = sampled_depth.squeeze() - proj_depth.squeeze()
    #     depth_diff_viz = depth_diff_viz.reshape(view_cur.image_height, view_cur.image_width)
    #     depth_diff_viz = torch.clamp(depth_diff_viz, -1, 1) + 1 / 2
    #     depth_diff_viz = cv2.applyColorMap((depth_diff_viz * 255).detach().cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)
    #     cv2.imwrite('debug/depth_diff.png', depth_diff_viz)

    proj_uvw_ = pts_neighbor_view / pts_neighbor_view[:, 2:3] * sampled_depth.unsqueeze(1)
    proj_uvw_homo = torch.concat([proj_uvw_, torch.ones_like(proj_uvw_[:, 0:1])], dim=-1)
    reproj_uvw = proj_uvw_homo @ (view_neighbor.world_view_transform_inv @ view_cur.world_view_transform)
    reproj_uvw = reproj_uvw[:, :3] @ view_cur.intrins
    reproj_uvw = reproj_uvw / reproj_uvw[: , 2:3] 
    pixel_noise = (pixels - reproj_uvw[:, :2]).norm(dim=-1)
    valid_mask = (pixel_noise < pixel_noise_threshold) & proj_uv_mask & (pts_cur['homo_plane_depth'] > 0)
    weights = (1.0 / torch.exp(pixel_noise)).detach()
    weights[~valid_mask] = 0

    # if debug:
    #     reproj_viz = np.zeros((view_cur.image_height, view_cur.image_width, 3 ), np.uint8)
    #     for i in range(0, reproj_uvw.shape[0], 100):
            
    #         pt0 = (int(pixels[i,0]), int(pixels[i,1]))
    #         pt1 = (int(reproj_uvw[i,0]), int(reproj_uvw[i,1]))
    #         cv2.circle(reproj_viz, pt0, 1, (255, 0, 0))
    #         cv2.circle(reproj_viz, pt1, 1, (0, 255, 0))
    #         cv2.line(reproj_viz, pt0, pt1, (0, 0, 255))
        
    #     # cv2.imshow("reproj_viz", reproj_viz)
    #     # cv2.waitKey()
    #     cv2.imwrite("debug/reproj.png", reproj_viz)

    if valid_mask.sum() > 0:
        loss_geo = torch.mean(weights[valid_mask] * pixel_noise[valid_mask])
        
        total_patch_size = patch_template.shape[1]

        with torch.no_grad():

            valid_indices = torch.arange(valid_mask.shape[0], device=valid_mask.device)[valid_mask]
            sample_num = 102_400 
            if valid_mask.sum() > sample_num:
                index = np.random.choice(valid_mask.sum().cpu().numpy(), sample_num, replace = False)
                valid_indices = valid_indices[index]

            ori_pixels_patch = patch_template.clone()[valid_indices]

            H, W = view_cur.gt_gray_img.squeeze().shape
            pixels_patch = ori_pixels_patch.clone()
            pixels_patch[:, :, 0] = 2 * pixels_patch[:, :, 0] / (W - 1) - 1.0
            pixels_patch[:, :, 1] = 2 * pixels_patch[:, :, 1] / (H - 1) - 1.0
            ref_gray_val = F.grid_sample(view_cur.gt_gray_img.unsqueeze(1), pixels_patch.view(1, -1, 1, 2), align_corners=True)
            ref_gray_val = ref_gray_val.reshape(-1, total_patch_size)

            ref_to_neareast_r = view_neighbor.world_view_transform[:3,:3].transpose(-1,-2) @ view_cur.world_view_transform[:3,:3]
            ref_to_neareast_t = -ref_to_neareast_r @ view_cur.world_view_transform[3,:3] + view_neighbor.world_view_transform[3,:3]
        
        # compute Homography
        ref_local_n = pts_cur['surf_normal_view'].reshape(-1, 3)[valid_indices]
        ref_local_d = pts_cur['homo_plane_depth'][valid_indices]

        H_ref_to_neareast = ref_to_neareast_r[None] - \
            torch.matmul(ref_to_neareast_t[None,:,None].expand(ref_local_d.shape[0],3,1), 
                        ref_local_n[:,:,None].expand(ref_local_d.shape[0],3,1).permute(0, 2, 1))/ref_local_d[...,None,None]
        H_ref_to_neareast = torch.matmul(view_neighbor.intrins.T[None].expand(ref_local_d.shape[0], 3, 3), H_ref_to_neareast)
        H_ref_to_neareast = H_ref_to_neareast @ view_cur.intrins_inv.T
        
        ## compute neareast frame patch
        grid = patch_warp(H_ref_to_neareast.reshape(-1,3,3), ori_pixels_patch)
        grid[:, :, 0] = 2 * grid[:, :, 0] / (W - 1) - 1.0
        grid[:, :, 1] = 2 * grid[:, :, 1] / (H - 1) - 1.0
        sampled_gray_val = F.grid_sample(
            view_neighbor.gt_gray_img.unsqueeze(0), 
            grid.reshape(1, -1, 1, 2), 
            align_corners=True
        )
        sampled_gray_val = sampled_gray_val.reshape(-1, total_patch_size)

        ncc, ncc_mask = lncc(ref_gray_val, sampled_gray_val)
        ncc = ncc.reshape(-1) * weights[valid_indices]
        loss_ncc = ncc[ncc_mask.reshape(-1)].mean() 
        ncc_map = torch.zeros_like(view_cur.gt_gray_img.squeeze()).view(-1).detach()
        ncc_map[valid_indices] = ncc
        ncc_map = ncc_map.reshape(H, W)

        loss_dict = {
            'geo': loss_geo,
            # 'geo': torch.tensor(0.0, device=valid_mask.device),
            'color': loss_ncc,
        }
        debug_dict = {
            'cur_patch': pixels_patch,
            'neighbor_patch': grid,
            'pixel_noise': pixel_noise.reshape(view_cur.image_height, view_cur.image_width),
            'sampled_depth': sampled_depth.reshape(view_cur.image_height, view_cur.image_width, 1),
            # 'view_neighbor_depth_diff': diff_depth.reshape(view_cur.image_height, view_cur.image_width, 1),
            'weights': weights,
            'ncc_map': ncc_map
        }
        return loss_dict, debug_dict

    else:
        loss_dict = {
            'geo': torch.tensor(0.0, device=valid_mask.device),
            'color': torch.tensor(0.0, device=valid_mask.device),
        }
        return loss_dict, None


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_dist_for_log = 0.0
    ema_normal_for_log = 0.0

    offsets_template_7 = patch_offsets(3, 'cuda')
    patch_template = None

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    viewpoint_stack_visit = 0

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_stack_visit += 1
            
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # if iteration % 100 == 0:
        if iteration >= opt.multiview_depth_iter:
            viewpoint_neigh_cam = scene.getTrainCameras()[sample(viewpoint_cam.nearest_id,1)[0]]
            render_pkg = render(viewpoint_cam, gaussians, pipe, background, return_pts=True)
            render_pkg_neigh = render(viewpoint_neigh_cam, gaussians, pipe, background, return_pts=True)

            W, H = viewpoint_cam.image_width, viewpoint_cam.image_height
            if patch_template is None:
                grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
                pixels = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2).float().cuda()
                patch_template = pixels.reshape(-1, 1, 2) + offsets_template_7.float()

            loss_dict, debug_dict = loss_in_neighbor_view(
                viewpoint_cam, viewpoint_neigh_cam, 
                render_pkg, render_pkg_neigh,
                patch_template, pixels, 1.0,
                debug=False
            )
            # loss_dict, debug_dict = loss_in_neighbor_view(
            #     viewpoint_cam, viewpoint_cam, 
            #     render_pkg, render_pkg,
            #     patch_template, pixels, 1.0,
            # )
            loss = opt.lambda_multiview_geo * loss_dict['geo'] + opt.lambda_multiview_ncc * loss_dict['color']
            # loss = 0
        
        else:
            render_pkg = render(viewpoint_cam, gaussians, pipe, background)
            loss = 0
            debug_dict = None
            loss_dict = {}

        image, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        gt_image = viewpoint_cam.original_image.cuda()
        loss_dict['Ll1'] = l1_loss(image, gt_image)
        loss_dict['ssim'] = 1 - ssim(image, gt_image)
        loss += (1.0 - opt.lambda_dssim) * loss_dict['Ll1'] + opt.lambda_dssim * loss_dict['ssim']

        # matching depth loss
        loss_dict['depth'] = ((render_pkg['surf_depth'] - viewpoint_cam.depth_aggregated).abs() * viewpoint_cam.depth_mask_aggregated).mean()
        if iteration > 1 and iteration % 2000 == 0:
            opt.lambda_depth *= 0.9
        if opt.lambda_depth > 0.0:
            loss += opt.lambda_depth * loss_dict['depth']

        # regularization
        lambda_normal = opt.lambda_normal if iteration > 3000 else 0.0
        lambda_dist = opt.lambda_dist if iteration > 1000 else 0.0

        rend_dist = render_pkg["rend_dist"]
        rend_normal  = render_pkg['rend_normal']
        surf_normal = render_pkg['surf_normal']
        normal_error = (1 - (rend_normal * surf_normal).sum(dim=0))[None]
        loss_dict['normal'] = lambda_normal * (normal_error).mean()
        loss_dict['distortion'] = lambda_dist * (rend_dist).mean()
 
        # loss
        total_loss = loss + loss_dict['normal'] + loss_dict['distortion']
        total_loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            ema_dist_for_log = 0.4 * loss_dict['distortion'].item() + 0.6 * ema_dist_for_log
            ema_normal_for_log = 0.4 * loss_dict['normal'].item() + 0.6 * ema_normal_for_log

            if iteration % 200 == 0:
                gt_img = tensor2cv(viewpoint_cam.original_image)
                render_img = tensor2cv(image)
                # far_plane = max(5, viewpoint_cam.depth_aggregated.max().item())
                # near_plane = viewpoint_cam.depth_aggregated.min().item()
                far_plane = 5
                near_plane = 0.1

                depth_corrected = apply_depth_colormap(
                    render_pkg['surf_depth'].squeeze(), 
                    near_plane=near_plane, far_plane=far_plane
                )
                sparse_depth = apply_depth_colormap(
                    viewpoint_cam.depth_aggregated.squeeze(),
                    near_plane=near_plane, far_plane=far_plane
                )
                normal_map = tensor2cv(render_pkg['surf_normal'].squeeze() / 2 + 0.5, permute=True)

                render_diff = viewpoint_cam.original_image - image
                render_diff = tensor2cv(render_diff.abs() * 5)

                depth_diff = torch.clamp((render_pkg['surf_depth'].squeeze() - viewpoint_cam.depth_aggregated) * 5 + 0.5, 0, 1) * 255
                depth_diff_img = cv2.applyColorMap(depth_diff.detach().cpu().numpy().astype(np.uint8), cv2.COLORMAP_TWILIGHT_SHIFTED)

                # empty_img = np.zeros_like(normal_map)
                pixel_depth_disorder_map = torch.clamp(render_pkg['pixel_depth_disorder'].squeeze(), 0, 255)
                pixel_depth_disorder_map = cv2.applyColorMap(pixel_depth_disorder_map.detach().cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)

                render_distortion_map = torch.clamp(rend_dist.squeeze() * 1000, 0, 1)  # H, W
                render_distortion_map = cv2.applyColorMap((render_distortion_map * 255).detach().cpu().numpy().astype(np.uint8), cv2.COLORMAP_JET)

                row1 = np.concatenate((gt_img, render_img, render_diff, render_distortion_map), axis=1)
                row2 = np.concatenate((sparse_depth, depth_corrected, depth_diff_img, normal_map), axis=1)

                if debug_dict is not None:
                    
                    # homo_plane_depth_viz = apply_depth_colormap(
                    #     render_pkg['homo_plane_depth_viz'].squeeze(),
                    #     near_plane=near_plane, far_plane=far_plane
                    # )
                    # surf_normal_viz = tensor2cv(
                    #     render_pkg['surf_normal'].squeeze() / 2 + 0.5, permute=True
                    # )
                    
                    ncc_map = (debug_dict['ncc_map']*255).detach().cpu().numpy().astype(np.uint8)
                    ncc_map = cv2.applyColorMap(ncc_map, cv2.COLORMAP_JET)

                    weights = (debug_dict['weights']*255).detach().cpu().numpy().astype(np.uint8).reshape(H,W)
                    weights_img = cv2.applyColorMap(weights, cv2.COLORMAP_JET)
                    row1 = np.concatenate(([row1, ncc_map]), axis=1)
                    row2 = np.concatenate(([row2, weights_img]), axis=1)
                    
                vis_img = np.concatenate((row1, row2), axis=0)
                vis_img = cv2.resize(vis_img, None, fx=0.5, fy=0.5)
                cv2.imwrite(os.path.join(scene.model_path, f'vis_{iteration:05d}.png'), vis_img)

            if iteration % 10 == 0:
                loss_dict_str = {
                    "Loss": f"{ema_loss_for_log:.{5}f}",
                    "distort": f"{ema_dist_for_log:.{5}f}",
                    "normal": f"{ema_normal_for_log:.{5}f}",
                    "Points": f"{len(gaussians.get_xyz)}"
                }
                progress_bar.set_postfix(loss_dict_str)

                progress_bar.update(10)

            if iteration == opt.iterations:
                progress_bar.close()


            training_report(tb_writer, iteration, loss_dict, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)


            # Densification
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, opt.opacity_cull, scene.cameras_extent, size_threshold)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")


def prepare_output_and_logger(args, exp_name = ''):
    if not args.model_path:
        # if os.getenv('OAR_JOB_ID'):
        #     unique_str=os.getenv('OAR_JOB_ID')
        # else:
        #     unique_str = str(uuid.uuid4())
        unique_str = exp_name + datetime.now().strftime("%Y%m%d_%H%M%S")
        args.model_path = os.path.join("./output/", unique_str)
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


@torch.no_grad()
def training_report(tb_writer, iteration, loss_dict, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        for k, v in loss_dict.items():
            tb_writer.add_scalar('train/' + k, v.item(), iteration)
        tb_writer.add_scalar('meta/iter_time', elapsed, iteration)
        tb_writer.add_scalar('metatotal_points', scene.gaussians.get_xyz.shape[0], iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0).to("cuda")
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar('eval_' + config['name'] + '/loss - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar('eval_' + config['name'] + '/loss - psnr', psnr_test, iteration)

        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[1_000, 2_000, 3_000, 4_000, 5_000, 6_000, 7_000, 8_000, 9_000, 10_000, 12_000, 15_000, 17_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[10_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint)

    # All done
    print("\nTraining complete.")
