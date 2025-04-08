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

import torch
import math
from torch import nn
import numpy as np
from torch.nn.functional import normalize
from utils.graphics_utils import getWorld2View2, getProjectionMatrix
from PIL import Image
from utils.general_utils import PILtoTorch, fov2focal


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.nearest_id = []
        self.nearest_names = []
        self.nearest_matched = []

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device) # move to device at dataloader to reduce VRAM requirement
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]
        self.depth_aggregated = np.zeros((self.image_height, self.image_width))
        self.valid_mask_aggregated = np.zeros((self.image_height, self.image_width))

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.gt_alpha_mask = gt_alpha_mask.to(self.data_device)
        else:
            # self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device) # do we need this?
            self.gt_alpha_mask = None
        
        self.zfar = 100.0
        self.znear = 0.01
        
        self.Fx = fov2focal(FoVx, self.image_width)
        self.Fy = fov2focal(FoVy, self.image_height)
        self.Cx = 0.5 * self.image_width
        self.Cy = 0.5 * self.image_height

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        self.world_view_transform_inv = torch.inverse(self.world_view_transform)

        fx = 1 / (2 * math.tan(self.FoVx / 2.))
        fy = 1 / (2 * math.tan(self.FoVy / 2.))
        self.intrins = torch.tensor(
            [[fx * self.image_width, 0., self.Cx],
            [0., fy * self.image_height, self.Cy],
            [0., 0., 1.0]],
            device=self.data_device
        ).float().T
        
        self.intrins_inv = torch.inverse(self.intrins)
        self.gt_gray_img = self.original_image.mean(0).unsqueeze(0)

        grid_x, grid_y = torch.meshgrid(torch.arange(self.image_width), torch.arange(self.image_height), indexing='xy')
        points = torch.stack([grid_x, grid_y, torch.ones_like(grid_x)], dim=-1).reshape(-1, 3).float().to(self.data_device)
        self.rays_d = points @ self.intrins_inv
        self.rays_d_normalized = normalize(self.rays_d, dim=-1)
        # self.rays_d_world_normalized = self.rays_d_normalized @ self.world_view_transform_inv[:3, :3]
        # self.rays_d_world_normalized = self.rays_d_world_normalized.reshape(self.image_height, self.image_width, 3)

        self.accm_depth = torch.zeros((self.image_height, self.image_width), device=self.data_device)
        self.accm_depth_conf = torch.zeros((self.image_height, self.image_width), device=self.data_device)

    def get_image(self):
        return self.original_image.cuda()
        # if self.preload_img:
        #     return self.original_image.cuda(), self.image_gray.cuda()
        # else:
        #     image = Image.open(self.image_path)
        #     resized_image = image.resize((self.image_width, self.image_height))
        #     resized_image_rgb = PILtoTorch(resized_image)
        #     if self.ncc_scale != 1.0:
        #         resized_image = image.resize((int(self.image_width/self.ncc_scale), int(self.image_height/self.ncc_scale)))
        #     resized_image_gray = resized_image.convert('L')
        #     resized_image_gray = PILtoTorch(resized_image_gray)
        #     gt_image = resized_image_rgb[:3, ...].clamp(0.0, 1.0)
        #     gt_image_gray = resized_image_gray.clamp(0.0, 1.0)
        #     return gt_image.cuda(), gt_image_gray.cuda()

    def get_calib_matrix_nerf(self, scale=1.0):
        intrinsic_matrix = torch.tensor([[self.Fx/scale, 0, self.Cx/scale], [0, self.Fy/scale, self.Cy/scale], [0, 0, 1]]).float()
        extrinsic_matrix = self.world_view_transform.transpose(0,1).contiguous() # cam2world
        return intrinsic_matrix, extrinsic_matrix
    
    def get_rays(self, scale=1.0):
        W, H = int(self.image_width/scale), int(self.image_height/scale)
        ix, iy = torch.meshgrid(
            torch.arange(W), torch.arange(H), indexing='xy')
        rays_d = torch.stack(
                    [(ix-self.Cx/scale) / self.Fx * scale,
                    (iy-self.Cy/scale) / self.Fy * scale,
                    torch.ones_like(ix)], -1).float().cuda()
        return rays_d

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

