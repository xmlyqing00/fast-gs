import torch
import numpy as np
import json
import cv2
from tqdm import tqdm, trange
from pathlib import Path
from torch.nn.functional import normalize, interpolate, pad, conv2d
from arguments import ModelParams
from copy import deepcopy
from scene.cameras import Camera
from submodules.EfficientLoFTR.src.loftr import LoFTR, full_default_cfg, opt_default_cfg, reparameter


def patch_offsets(h_patch_size, device):
    offsets = torch.arange(-h_patch_size, h_patch_size + 1, device=device)
    return torch.stack(torch.meshgrid(offsets, offsets)[::-1], dim=-1).view(1, -1, 2)

patch_size = 2
offsets_template = patch_offsets(patch_size, 'cuda')
patch_weights = torch.exp(-torch.norm(offsets_template.float(), dim=-1) * 3)
patch_weights = patch_weights.view(1, 1, 2 * patch_size + 1, 2 * patch_size + 1)


def build_neighbors(cameras: list, args: ModelParams):
    
    print('Computing nearest_id')
    camera_centers = []
    center_rays = []
    for cam in cameras:
        camera_centers.append(cam.camera_center)
        R = torch.tensor(cam.R).float().cuda()
        center_ray = torch.tensor([0.0,0.0,1.0]).float().cuda()
        center_ray = center_ray @ R.transpose(-1,-2)
        center_rays.append(center_ray)
    camera_centers = torch.stack(camera_centers, dim=0)
    center_rays = torch.stack(center_rays, dim=0)
    center_rays = normalize(center_rays, dim=-1)
    diss = torch.norm(camera_centers[:,None] - camera_centers[None], dim=-1).detach().cpu().numpy()
    tmp = torch.sum(center_rays[:,None] * center_rays[None], dim=-1)
    angles = torch.arccos(tmp) * 180 / 3.1415926
    angles = angles.detach().cpu().numpy()
    model_path = Path(args.model_path)
    with open(model_path / 'multi_view.json', 'w') as file:
        for id, cam in enumerate(cameras):
            sorted_indices = np.lexsort((angles[id], diss[id]))
            mask = (angles[id][sorted_indices] < args.multi_view_max_angle) & \
                (diss[id][sorted_indices] > args.multi_view_min_dis) & \
                (diss[id][sorted_indices] < args.multi_view_max_dis)
            sorted_indices = sorted_indices[mask]
            multi_view_num = min(args.multi_view_num, len(sorted_indices))
            json_d = {'ref_name' : cam.image_name, 'nearest_name': []}
            for index in sorted_indices[:multi_view_num]:
                cam.nearest_id.append(index.item())
                cam.nearest_names.append(cameras[index].image_name)
                json_d['nearest_name'].append(cameras[index].image_name)
            json_str = json.dumps(json_d, separators=(',', ':'))
            file.write(json_str)
            file.write('\n')


def build_matching_module(
        precision: str, 
        model_type: str = 'full', 
        pretrain_weights: str = 'weights/eloftr_outdoor.ckpt'
    ) -> LoFTR:

    # You can choose model type in ['full', 'opt']
    assert model_type in ['full', 'opt']
    # model_type = 'full' # 'full' for best quality, 'opt' for best efficiency

    # You can choose numerical precision in ['fp32', 'mp', 'fp16']. 'fp16' for best efficiency
    assert precision in ['fp16', 'mp', 'fp32']
    # precision = 'fp16' # Enjoy near-lossless precision with Mixed Precision (MP) / FP16 computation if you have a modern GPU (recommended NVIDIA architecture >= SM_70).

    # You can also change the default values like thr. and npe (based on input image size)
    if model_type == 'full':
        _default_cfg = deepcopy(full_default_cfg)
    elif model_type == 'opt':
        _default_cfg = deepcopy(opt_default_cfg)
        
    if precision == 'mp':
        _default_cfg['mp'] = True
    elif precision == 'fp16':
        _default_cfg['half'] = True

    matcher = LoFTR(config=_default_cfg)

    matcher.load_state_dict(torch.load(pretrain_weights, weights_only=False)['state_dict'])
    matcher = reparameter(matcher) # no reparameterization will lead to low performance
    
    if precision == 'fp16':
        matcher = matcher.half()

    matcher = matcher.eval().cuda()
    return matcher
    

def draw_epipolar_errors(target_img, source_img, pts1, pts2, viz_data, epipolar_thresh=1.0):
    """
    Draw epipolar lines and errors on the images.

    Args:
        target_img: Target image (H,W,3)
        source_img: Source image (H,W,3)
        pts1, pts2: Original point correspondences
        viz_data: Visualization data from triangulate_to_depth_map
        epipolar_thresh: Threshold used for epipolar error
    """
    import matplotlib.pyplot as plt

    # Create figure
    plt.figure(figsize=(15, 10))

    # Plot first image with epipolar lines
    plt.subplot(221)
    plt.imshow(target_img)
    plt.title('Target Image with Epipolar Lines')

    # Draw epipolar lines in first image
    for i in range(len(pts1)):
        color = 'g' if viz_data['epipolar_errors'][i] < epipolar_thresh else 'r'
        line = viz_data['epipolar_lines1'][i]
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [target_img.shape[1], -(line[2] + line[0] * target_img.shape[1]) / line[1]])
        plt.plot([x0, x1], [y0, y1], color, alpha=0.5)
        plt.plot(pts1[i, 0], pts1[i, 1], color + '.')

    # Plot second image with epipolar lines
    plt.subplot(222)
    plt.imshow(source_img)
    plt.title('Source Image with Epipolar Lines')

    # Draw epipolar lines in second image
    for i in range(len(pts2)):
        color = 'g' if viz_data['epipolar_errors'][i] < epipolar_thresh else 'r'
        line = viz_data['epipolar_lines2'][i]
        x0, y0 = map(int, [0, -line[2] / line[1]])
        x1, y1 = map(int, [source_img.shape[1], -(line[2] + line[0] * source_img.shape[1]) / line[1]])
        plt.plot([x0, x1], [y0, y1], color, alpha=0.5)
        plt.plot(pts2[i, 0], pts2[i, 1], color + '.')

    # Plot error histogram
    plt.subplot(223)
    plt.hist(viz_data['epipolar_errors'], bins=50)
    plt.axvline(epipolar_thresh, color='r', linestyle='--', label=f'Threshold ({epipolar_thresh}px)')
    plt.xlabel('Epipolar Error (pixels)')
    plt.ylabel('Count')
    plt.title('Distribution of Epipolar Errors')
    plt.legend()
    plt.grid(True)

    # Plot depth histogram
    plt.subplot(224)
    valid_depths = viz_data['depths'][viz_data['depth_mask']]
    plt.hist(valid_depths, bins=50)
    plt.xlabel('Depth (units)')
    plt.ylabel('Count')
    plt.title('Distribution of Valid Depths')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    # Print statistics
    print(f"Mean epipolar error: {viz_data['epipolar_errors'].mean():.3f} pixels")
    print(f"Median epipolar error: {np.median(viz_data['epipolar_errors']):.3f} pixels")
    print(f"Max epipolar error: {viz_data['epipolar_errors'].max():.3f} pixels")
    print(
        f"Epipolar inliers: {viz_data['epipolar_mask'].sum()}/{len(pts1)} ({100 * viz_data['epipolar_mask'].mean():.1f}%)")
    print(f"Depth inliers: {viz_data['depth_mask'].sum()}/{len(pts1)} ({100 * viz_data['depth_mask'].mean():.1f}%)")


def build_sparse_depth(
        depths: np.array, mconf: np.array, epipolar_mask: np.array, pts: np.array,
        depth_range: tuple, img_shape: tuple
    ) -> tuple[np.array, np.array]:

    # Combine all filters
    depth_mask = (depths > depth_range[0]) & (depths < depth_range[1])  # Adjust max depth as needed
    inlier_mask = depth_mask & epipolar_mask

    if not np.any(inlier_mask):
        # print("[Warning] No inliers found!")
        return (
            np.zeros(img_shape),
            np.zeros(img_shape),
        )

    # Create sparse depth map
    sparse_depth = np.zeros(img_shape)
    valid_mask = np.zeros(img_shape)

    # Convert points back to pixel coordinates for the depth map
    pts_inliers = pts[inlier_mask]
    depths_inliers = depths[inlier_mask]
    mconf_inliers = mconf[inlier_mask]

    # Round pixel coordinates and ensure they're within image bounds
    pts1_px = np.round(pts_inliers).astype(int)
    valid_pts = (pts1_px[:, 0] >= 0) & (pts1_px[:, 0] < img_shape[1]) & \
                (pts1_px[:, 1] >= 0) & (pts1_px[:, 1] < img_shape[0])

    # Fill in the sparse depth map
    pts1_px = pts1_px[valid_pts]
    depths_inliers = depths_inliers[valid_pts]
    mconf_inliers = mconf_inliers[valid_pts]

    sparse_depth[pts1_px[:, 1], pts1_px[:, 0]] = depths_inliers
    valid_mask[pts1_px[:, 1], pts1_px[:, 0]] = mconf_inliers

    return sparse_depth, valid_mask


def triangulate_to_depth_map(
        target_img_shape, 
        pts1, pts2, mconf, 
        target_pose, source_pose, K, epipolar_thresh
    ):
    """
    Triangulate matched points to create a sparse depth map with epipolar error filtering.
    Also returns visualization data for epipolar geometry.
    Credit: https://github.com/denyingmxd/simplerecon/blob/bdb53f6c411b5c47dfbd23eecc9946d0d4c3bc24/get_sparse_depth.py#L139
    """
    if pts1 is None or pts2 is None or len(pts1) == 0 or len(pts2) == 0 :

        return (
            np.zeros(target_img_shape),
            np.zeros(target_img_shape, dtype=bool),
            np.zeros(target_img_shape),
            np.zeros(target_img_shape, dtype=bool),
        )

    # Calculate relative pose between cameras
    # source_pose and target_pose are world_T_cam
    # relative_pose = w2c(source) @ c2w(target)
    relative_pose = np.linalg.inv(source_pose) @ target_pose  # source_cam_T_target_cam

    # Extract R and t from relative pose
    R = relative_pose[:3, :3]
    t = relative_pose[:3, 3]

    # Calculate essential matrix
    t_cross = np.array([
        [0, -t[2], t[1]],
        [t[2], 0, -t[0]],
        [-t[1], t[0], 0]
    ])
    E = t_cross @ R

    # Calculate fundamental matrix
    F = np.linalg.inv(K[:3, :3]).T @ E @ np.linalg.inv(K[:3, :3])

    # Reshape points for OpenCV function
    pts1_reshaped = pts1.reshape(-1, 1, 2)
    pts2_reshaped = pts2.reshape(-1, 1, 2)

    # Compute epipolar lines using OpenCV
    lines1 = cv2.computeCorrespondEpilines(pts2_reshaped, 2, F)
    lines2 = cv2.computeCorrespondEpilines(pts1_reshaped, 1, F)

    # Reshape lines
    lines1 = lines1.reshape(-1, 3)
    lines2 = lines2.reshape(-1, 3)

    # Calculate epipolar errors
    pts1_h = np.hstack([pts1, np.ones((len(pts1), 1))])
    pts2_h = np.hstack([pts2, np.ones((len(pts2), 1))])

    # Distance from points to epipolar lines
    errors1 = np.abs(np.sum(pts1_h * lines1, axis=1)) / \
              np.sqrt(lines1[:, 0] ** 2 + lines1[:, 1] ** 2)
    errors2 = np.abs(np.sum(pts2_h * lines2, axis=1)) / \
              np.sqrt(lines2[:, 0] ** 2 + lines2[:, 1] ** 2)

    # Average symmetric epipolar distance
    epipolar_errors = (errors1 + errors2) / 2

    # Filter based on epipolar error
    epipolar_mask = epipolar_errors < epipolar_thresh

    # Get projection matrices
    P1 = K[:3, :3] @ np.hstack([np.eye(3), np.zeros((3, 1))])  # target camera
    P2 = K[:3, :3] @ np.hstack([R, t.reshape(3, 1)])  # source camera

    # Triangulate points
    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts4d /= pts4d[3]
    pts3d = pts4d[:3].T

    # The points are already in target camera coordinate frame
    # Just extract Z coordinates as depths
    depths = pts3d[:, 2]
    sparse_depth_target, valid_mask_target = build_sparse_depth(
        depths, mconf, epipolar_mask, pts1, 
        (0, 10), target_img_shape
    )

    # project the pts3d to source camera view
    pts_4d_source = P2 @ pts4d
    depths_source = pts_4d_source[2]
    sparse_depth_source, valid_mask_source = build_sparse_depth(
        depths_source, mconf, epipolar_mask, pts2, 
        (0, 10), target_img_shape
    )

    return sparse_depth_target, valid_mask_target, sparse_depth_source, valid_mask_source



def interpolate_depth(depth):
    mask = (depth > 0).astype(np.uint8)  # Non-zero mask
    return cv2.inpaint(depth.astype(np.float32), mask, inpaintRadius=5, flags=cv2.INPAINT_TELEA)

 
def estimate_normal_map(depth: torch.tensor, view: Camera) -> torch.tensor:

    # interpolate depth by gaussian filter
    # depth = conv2d(depth[None, None, ...], patch_weights, padding=2)[0, 0]
    # weight = conv2d((depth > 0).float()[None, None, ...], patch_weights, padding=2)[0, 0]
    # depth = depth / weight.clamp(min=1e-6)

    # inverse project pixels to 3D points
    xyz_view = depth.view(-1, 1) * view.rays_d
    xyz_world = xyz_view @ view.world_view_transform_inv[:3, :3] + view.world_view_transform_inv[3, :3]

    # compute normal
    points = xyz_world.reshape(view.image_height, view.image_width, 3)
    dx = points[2:, 1:-1] - points[:-2, 1:-1]
    dy = points[1:-1, 2:] - points[1:-1, :-2]
    normal_map = torch.cross(dx, dy, dim=-1)
    normal_map = normalize(normal_map, dim=-1)  # H, W, 3
    normal_map = pad(normal_map.permute(2, 0, 1), (1, 1, 1, 1), mode='replicate').permute(1, 2, 0)  # pad for boundary (2, 2, 0)
    return normal_map


def est_depth(cameras: list, args: ModelParams):

    # Encode the frame
    n = len(cameras)
    chunk_size = 32
    chunks = n // chunk_size
    if n % chunk_size:
        chunks += 1

    matcher = build_matching_module(args.precision)
    hw0_i = None

    # Encode the features
    for i in trange(chunks, desc='Encoding features'):
        print(f"Estimating depth for chunk {i}")
        imgs = []
        img_num = min(chunk_size, n - i * chunk_size)
        for j in range(img_num):
            img = cameras[i*chunk_size+j].original_image
            img_gray = 0.2989 * img[0] + 0.5870 * img[1] + 0.1140 * img[2]
            img_gray = interpolate(img_gray[None, None, ...], (img_gray.shape[0]//32*32, img_gray.shape[1]//32*32), mode='bilinear', align_corners=False)
            cameras[i*chunk_size+j].resize_ratio = np.array(
                [[img.shape[1] / img_gray.shape[2], img.shape[2] / img_gray.shape[3]]]
            )
            # cameras[i*chunk_size+j].img_gray = img_gray.squeeze()
            if i == 0 and j == 0:
                hw0_i = img_gray.shape[2:]
            imgs.append(img_gray)
        imgs = torch.concat(imgs, dim=0)
        if args.precision == 'fp16':
            imgs = imgs.half()

        with torch.no_grad():
            if args.precision == 'mp':
                with torch.autocast(enabled=True, device_type='cuda'):
                    ret_dict = matcher.backbone(imgs)
            else:
                ret_dict = matcher.backbone(imgs)

            for j in range(img_num):
                cameras[i*chunk_size+j].feats = {
                    'feats_x2': ret_dict['feats_x2'][j].unsqueeze(0),
                    'feats_x1': ret_dict['feats_x1'][j].unsqueeze(0),
                    'feats_c': ret_dict['feats_c'][j].unsqueeze(0)
                }

    # Match the features
    mul = matcher.config['resolution'][0] // matcher.config['resolution'][1]
    data_shape = {
        'bs': 1,
        'hw0_i': hw0_i, 'hw1_i': hw0_i,
        'hw0_c': cameras[0].feats['feats_c'].shape[2:], 
        'hw1_c': cameras[0].feats['feats_c'].shape[2:],
        'hw0_f': [cameras[0].feats['feats_c'].shape[2] * mul, cameras[0].feats['feats_c'].shape[3] * mul],
        'hw1_f': [cameras[0].feats['feats_c'].shape[2] * mul, cameras[0].feats['feats_c'].shape[3] * mul]
    }

    for cam_id, cam in tqdm(enumerate(cameras), desc='Matching for depth'):
        
        cam = cameras[cam_id]

        for neighbor_id in cam.nearest_id:
            if neighbor_id in cam.nearest_matched:
                # print(f"Skip matching between {cam_id} and {neighbor_id}")
                continue

            cam_neighbor = cameras[neighbor_id]
            cam.nearest_matched.append(neighbor_id)
            cam_neighbor.nearest_matched.append(cam_id)

            data = {
                'feats_x1': torch.cat([cam.feats['feats_x1'], cam_neighbor.feats['feats_x1']], dim=0),
                'feats_x2': torch.cat([cam.feats['feats_x2'], cam_neighbor.feats['feats_x2']], dim=0),
                'feats_c0': cam.feats['feats_c'],
                'feats_c1': cam_neighbor.feats['feats_c'],
            }
            data.update(data_shape)
            with torch.no_grad():
                if args.precision == 'mp':
                    with torch.autocast(enabled=True, device_type='cuda'):
                        matcher.forward_pair(data)
                else:
                    matcher.forward_pair(data)
            
            matching_mask = data['mconf'] > args.matching_thresh
            mkpts0 = data['mkpts0_f'][matching_mask].cpu().numpy()
            mkpts0 = mkpts0 * cam.resize_ratio
            mkpts1 = data['mkpts1_f'][matching_mask].cpu().numpy()
            mkpts1 = mkpts1 * cam_neighbor.resize_ratio
            mconf = data['mconf'][matching_mask].cpu().numpy()

            # remove corner points
            # corner_width = int(min(cam.image_width, cam.image_height) * 0.1)
            # corner_mask = (mkpts0[:, 0] > corner_width) & (mkpts0[:, 0] < cam.image_width - corner_width) & \
            #     (mkpts0[:, 1] > corner_width) & (mkpts0[:, 1] < cam.image_height - corner_width)
            # mkpts0 = mkpts0[corner_mask]
            # mkpts1 = mkpts1[corner_mask]
            # mconf = mconf[corner_mask]

            sparse_depth_target, valid_mask_target, sparse_depth_source, valid_mask_source = triangulate_to_depth_map(
                (cam.image_height, cam.image_width),
                mkpts0,
                mkpts1,
                mconf,
                cam.world_view_transform_inv.T.cpu().numpy(),
                cam_neighbor.world_view_transform_inv.T.cpu().numpy(),
                cam.intrins.T.cpu().numpy(),
                args.epipolar_thresh
            )

            # convert depth to jet colormap 
            # sparse_depth_viz = sparse_depth_target.copy() / sparse_depth_target.max()
            # sparse_depth_viz = np.clip(sparse_depth_viz, 0, 1)
            # sparse_depth_viz = cv2.applyColorMap((sparse_depth_viz * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # cv2.imshow('sparse_depth_target', sparse_depth_viz)
            # cv2.imshow('valid_mask_target', valid_mask_target.astype(np.float64))
            # cv2.imshow('img_target', cam.img_gray.cpu().numpy())

            # sparse_depth_source_viz = sparse_depth_source.copy() / sparse_depth_source.max()
            # sparse_depth_source_viz = np.clip(sparse_depth_source_viz, 0, 1)
            # sparse_depth_source_viz = cv2.applyColorMap((sparse_depth_source_viz * 255).astype(np.uint8), cv2.COLORMAP_JET)
            # cv2.imshow('sparse_depth_source', sparse_depth_source_viz)
            # cv2.imshow('valid_mask_source', valid_mask_source.astype(np.float64))
            # cv2.imshow('img_source', cam_neighbor.img_gray.cpu().numpy())
            # cv2.waitKey(0)

            # aggregate depth and valid mask
            cam.depth_aggregated += sparse_depth_target * valid_mask_target
            cam.valid_mask_aggregated += valid_mask_target
            cam_neighbor.depth_aggregated += sparse_depth_source * valid_mask_source
            cam_neighbor.valid_mask_aggregated += valid_mask_source
    

    debug_dir = Path(args.model_path) / 'debug'
    debug_dir.mkdir(parents=True, exist_ok=True)
    
    for cam_id, cam in tqdm(enumerate(cameras), desc='Aggregating depth'):

        # if cam.image_name == '0024':
            # print('debug')
        # normalize depth
        cam.depth_aggregated = cam.depth_aggregated / np.maximum(1e-6, cam.valid_mask_aggregated)
        # interpolate depth
        # cam.depth_aggregated = interpolate_depth(cam.depth_aggregated)
        
        cam.depth_mask_aggregated = cam.valid_mask_aggregated > 0

        cam.depth_aggregated = torch.from_numpy(cam.depth_aggregated).float().cuda()
        cam.depth_mask_aggregated = torch.from_numpy(cam.depth_mask_aggregated).float().cuda()

        # depth = conv2d(cam.depth_aggregated[None, None, ...], patch_weights, padding=patch_size)[0, 0]
        # weight = conv2d(cam.depth_mask_aggregated[None, None, ...], patch_weights, padding=patch_size)[0, 0]
        # cam.depth_aggregated = depth / weight.clamp(min=1e-6)
        # cam.depth_mask_aggregated = cam.depth_aggregated > 0
        # # cam.normal_mask_aggregated = cam.depth_mask_aggregated

        # convert depth to jet colormap 
        sparse_depth_viz = cam.depth_aggregated.cpu().numpy().copy() / 4
        # depth_normal_valid_mask = sparse_depth_viz + cam.normal_mask_aggregated.cpu().numpy()
        # depth_normal_valid_mask = np.clip(depth_normal_valid_mask, 0, 1)
        sparse_depth_viz = np.clip(sparse_depth_viz, 0, 1)
        sparse_depth_viz = cv2.applyColorMap((sparse_depth_viz * 255).astype(np.uint8), cv2.COLORMAP_JET)

        cv2.imwrite(debug_dir / f'{cam.image_name}_depth.png', sparse_depth_viz)
        cv2.imwrite(debug_dir / f'{cam.image_name}_depth_valid_mask.png', (cam.depth_mask_aggregated.cpu().numpy() * 255).astype(np.uint8))
        # cv2.imwrite(debug_dir / f'{cam.image_name}_normal_valid_mask.png', (cam.normal_mask_aggregated.cpu().numpy() * 255).astype(np.uint8))
        # cv2.imwrite(debug_dir / f'{cam.image_name}_depth_normal_mask.png', (depth_normal_valid_mask * 255).astype(np.uint8))
        
        # estimate normal map
        # cam.normal_map = estimate_normal_map(cam.depth_aggregated, cam)
        # # cam.normal_map = cam.normal_map + cam.normal_mask_aggregated[..., None]
        # cam.normal_map = cam.normal_map * cam.normal_mask_aggregated[..., None]
        # normal_map_viz = cam.normal_map.cpu().numpy()
        # normal_map_viz = (normal_map_viz + 1) / 2
        # normal_map_viz = (normal_map_viz * 255).astype(np.uint8)
        # cv2.imwrite(debug_dir / f'{cam.image_name}_normal_map.png', normal_map_viz)
        # cam.normal_map = cam.normal_map.permute(2, 0, 1)

        # draw_epipolar_errors(
        #     cam.img_gray.cpu().numpy(), 
        #     cam_neighbor.img_gray.cpu().numpy(), 
        #     mkpts0, mkpts1, 
        #     viz_data, 
        #     args.epipolar_thresh
        # )