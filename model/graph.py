import numpy as np
import math
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as torch_F
from utils.util import EasyDict as edict

from . import loss
from model.renderer import Renderer
from model.view_estimator import Estimator
from model.implicit import SDFNetwork, RGBNetwork
import utils.camera as camera
from utils.util import log

class Bottleneck_Linear(nn.Module):

    def __init__(self, n_channels):
        super().__init__()
        self.linear1 = nn.Conv2d(n_channels, n_channels, kernel_size=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(n_channels)
        self.linear2 = nn.Conv2d(n_channels, n_channels, kernel_size=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(n_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.unsqueeze(-1).unsqueeze(-1)
        residual = x

        out = self.linear1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.linear2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)
        out = out.squeeze(-1).squeeze(-1)

        return out

class Graph(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.estimator = Estimator(opt)
        self.sdf_network = SDFNetwork(opt)
        self.rgb_network = RGBNetwork(opt)
        self.renderer = Renderer(opt, self.sdf_network, self.rgb_network)
        network = getattr(torchvision.models, opt.arch.enc_network)
        self.encoder = network(pretrained=opt.arch.enc_pretrained)
        self.encoder.fc = nn.Linear(
            self.encoder.fc.in_features, 
            opt.arch.latent_dim_shape + opt.arch.latent_dim_rgb
        )
        self.latent_proj_shape = nn.Sequential(
            Bottleneck_Linear(opt.arch.latent_dim_shape),
            Bottleneck_Linear(opt.arch.latent_dim_shape),
            nn.Linear(opt.arch.latent_dim_shape, opt.arch.impl_sdf.proj_latent_dim)
        )
        self.latent_proj_rgb = nn.Sequential(
            Bottleneck_Linear(opt.arch.latent_dim_rgb),
            Bottleneck_Linear(opt.arch.latent_dim_rgb),
            nn.Linear(opt.arch.latent_dim_rgb, opt.arch.impl_rgb.proj_latent_dim)
        )
        self.loss_fns = loss.Loss(opt)
    
    def forward(self, opt, var, training=False, get_loss=True, visualize=False):
        batch_size = len(var.idx)
        ray_idx = var.ray_idx if (opt.render.rand_sample and training) else None

        # forward the encoder and reconstructor to get the shape and rgb volume
        var.latent_raw = var.latent if "latent" in var else self.encoder(var.rgb_input_map)
        var.latent_shape = var.latent_raw[:, :opt.arch.latent_dim_shape]
        var.latent_rgb = var.latent_raw[:, opt.arch.latent_dim_shape:]

        # get the feature for impl conditioning
        var.proj_latent_sdf = self.latent_proj_shape(var.latent_shape)
        var.proj_latent_rgb = self.latent_proj_rgb(var.latent_rgb)

        # predict the viewpoint
        var.pose, var.intr, var.scale_dist = self.pred_pose(opt, var)  
        
        # canonicalize the normal map
        var.normal_transformed = camera.transform_normal(var.normal_gt if 'normal_gt' in var else var.normal_input, var.pose)

        # render the image, save rendering results if needed
        if visualize:
            var.rgb_recon, var.mask_recon, var.mask_hard, var.depth_recon, var.normal_recon, var.grad_eikonal, var.rendering_points, var.rendering_transparency, var.rendering_rgb = \
                self.renderer(opt, var.pose, var.intr, var.scale_dist, var.proj_latent_sdf, var.proj_latent_rgb, ray_idx=ray_idx, training=training, visualize=visualize)
        else:
            var.rgb_recon, var.mask_recon, var.mask_hard, var.depth_recon, var.normal_recon, var.grad_eikonal = \
                self.renderer(opt, var.pose, var.intr, var.scale_dist, var.proj_latent_sdf, var.proj_latent_rgb, ray_idx=ray_idx, training=training)

        # reshape the outputs if needed
        if not (opt.render.rand_sample and training):
            var.rgb_recon_map = var.rgb_recon.view(batch_size, opt.H, opt.W, 3).permute(0, 3, 1, 2).contiguous()
            var.mask_recon_map = var.mask_recon.view(batch_size, opt.H, opt.W, 1).permute(0, 3, 1, 2).contiguous()
            var.mask_hard_map = var.mask_hard.view(batch_size, opt.H, opt.W, 1).permute(0, 3, 1, 2).contiguous()
            var.normal_recon_map = var.normal_recon.view(batch_size, opt.H, opt.W, 3).permute(0, 3, 1, 2).contiguous()
            var.normal_transformed_map = var.normal_transformed.view(batch_size, opt.image_size[0], opt.image_size[1], 3).permute(0, 3, 1, 2).contiguous()
        
        # forward for the NN if needed
        if (opt.loss_weight.nearest_img is not None or opt.loss_weight.nearest_mask is not None) and training:
            self.forward_NN(opt, var)
        
        # calculate the loss if needed
        if get_loss: 
            loss = self.compute_loss(opt, var, training)
            return var, loss
        
        return var

    def forward_NN(self, opt, var, training=True):
        batch_size = len(var.idx)
        assert opt.reg.n_views <= opt.data.k_nearest

        # select the best indices based on the viewpoint discrepancy
        with torch.no_grad():

            ious = []
            for i in range(opt.data.k_nearest):
                current_mask = var.mask_input_NN[...,i].view(batch_size, -1)
                input_mask = var.mask_input.view(batch_size, -1)
                iou = (current_mask * input_mask).sum(dim=1) / (current_mask + input_mask - current_mask * input_mask + 1.e-8).sum(dim=1)
                ious.append(iou)
            # [B, K]
            scores = 1 - torch.stack(ious, dim=-1)
            # scale the scores first for sharper weights
            scores = scores ** opt.reg.sample_temp
            # generate the index with weighted sampling
            probs = torch_F.normalize(scores, dim=-1, p=1)
            indices = []
            for i in range(batch_size):
                # [K]
                prob = probs[i].cpu().numpy()
                # normalize again for better numerical stability
                prob = prob / np.sum(prob)
                current_index = np.random.choice(opt.data.k_nearest, size=(opt.reg.n_views, ), replace=False, p=prob)
                indices.append(current_index)
                
            idx_NN = torch.tensor(np.stack(indices, axis=0)).long().to(var.rgb_input_map.device)
            # [B, V] -> [B, N, C, V]
            idx_NN_rgb = idx_NN.view(batch_size, 1, 1, opt.reg.n_views).expand(
                batch_size, var.rgb_input.shape[1], var.rgb_input.shape[2], -1
            )
            idx_NN_mask = idx_NN.view(batch_size, 1, 1, opt.reg.n_views).expand(
                batch_size, var.mask_input.shape[1], var.mask_input.shape[2], -1
            )
            idx_NN_normal = idx_NN.view(batch_size, 1, 1, opt.reg.n_views).expand(
                batch_size, var.normal_input.shape[1], var.normal_input.shape[2], -1
            )
            idx_NN_pose = idx_NN.view(batch_size, 1, 1, opt.reg.n_views).expand(
                batch_size, var.pose_gt.shape[1], var.pose_gt.shape[2], -1
            )
            # [B, V] -> [B, C, H, W, V]
            idx_NN_rgb_map = idx_NN.view(batch_size, 1, 1, 1, opt.reg.n_views).expand(
                batch_size, var.rgb_input_map.shape[1], var.rgb_input_map.shape[2], var.rgb_input_map.shape[3], -1
            )
            idx_NN_mask_map = idx_NN.view(batch_size, 1, 1, 1, opt.reg.n_views).expand(
                batch_size, var.mask_input_map.shape[1], var.mask_input_map.shape[2], var.mask_input_map.shape[3], -1
            )
            idx_NN_normal_map = idx_NN.view(batch_size, 1, 1, 1, opt.reg.n_views).expand(
                batch_size, var.normal_input_map.shape[1], var.normal_input_map.shape[2], var.normal_input_map.shape[3], -1
            )
            if opt.render.rand_sample and training:
                assert len(var.ray_idx.shape) == 2
                # [B, V] -> [B, N, V]
                idx_NN_ray_idx = idx_NN.view(batch_size, 1, opt.reg.n_views).expand(
                    batch_size, var.ray_idx.shape[1], -1
                )

        # forward for each NN input
        for view_id in range(opt.reg.n_views):
            # get the selected inputs
            var['input_NN_{}'.format(view_id)] = edict()
            var['input_NN_{}'.format(view_id)].rgb_input_map = \
                torch.gather(var.rgb_input_map_NN, -1, idx_NN_rgb_map[..., view_id:view_id+1]).squeeze(-1)
            var['input_NN_{}'.format(view_id)].mask_input_map = \
                torch.gather(var.mask_input_map_NN, -1, idx_NN_mask_map[..., view_id:view_id+1]).squeeze(-1)
            var['input_NN_{}'.format(view_id)].normal_input_map = \
                torch.gather(var.normal_input_map_NN, -1, idx_NN_normal_map[..., view_id:view_id+1]).squeeze(-1)
            var['input_NN_{}'.format(view_id)].rgb_input = \
                torch.gather(var.rgb_input_NN, -1, idx_NN_rgb[..., view_id:view_id+1]).squeeze(-1)
            var['input_NN_{}'.format(view_id)].mask_input = \
                torch.gather(var.mask_input_NN, -1, idx_NN_mask[..., view_id:view_id+1]).squeeze(-1)
            var['input_NN_{}'.format(view_id)].normal_input = \
                torch.gather(var.normal_input_NN, -1, idx_NN_normal[..., view_id:view_id+1]).squeeze(-1)
            if opt.render.rand_sample and training:
                var['input_NN_{}'.format(view_id)].ray_idx = \
                    torch.gather(var.ray_idx_NN, -1, idx_NN_ray_idx[..., view_id:view_id+1]).squeeze(-1)
            var['input_NN_{}'.format(view_id)].pose_gt = \
                torch.gather(var.pose_gt_NN, -1, idx_NN_pose[..., view_id:view_id+1]).squeeze(-1)
            ray_idx = var['input_NN_{}'.format(view_id)].ray_idx if opt.render.rand_sample and training else None

            # forward the encoder for NN rgb latent code
            latent_NN = self.encoder(var['input_NN_{}'.format(view_id)].rgb_input_map)
            latent_rgb_NN = latent_NN[:, opt.arch.latent_dim_shape:]
            proj_latent_rgb_NN = self.latent_proj_rgb(latent_rgb_NN)
            var.proj_latent_rgb_NN = proj_latent_rgb_NN

            # predict the pose
            var['pose_NN_{}'.format(view_id)], var['intr_NN_{}'.format(view_id)], var['scale_dist_NN_{}'.format(view_id)] = \
                self.pred_pose(opt, var, pred_NN=True, given_input=var['input_NN_{}'.format(view_id)].rgb_input_map)
            
            # render the NN recons
            var['rgb_recon_NN_{}'.format(view_id)], var['mask_recon_NN_{}'.format(view_id)], _, var['depth_recon_NN_{}'.format(view_id)], var['normal_recon_NN_{}'.format(view_id)], _ = \
                self.renderer(opt, var['pose_NN_{}'.format(view_id)], var['intr_NN_{}'.format(view_id)], var['scale_dist_NN_{}'.format(view_id)], 
                              var.proj_latent_sdf, proj_latent_rgb_NN, ray_idx=ray_idx, training=training)
            
            # reshape if needed
            if not (opt.render.rand_sample and training):
                var['rgb_recon_map_NN_{}'.format(view_id)] = \
                    var['rgb_recon_NN_{}'.format(view_id)].view(batch_size, opt.H, opt.W, 3).permute(0, 3, 1, 2).contiguous()
                var['mask_recon_map_NN_{}'.format(view_id)] = \
                    var['mask_recon_NN_{}'.format(view_id)].view(batch_size, opt.H, opt.W, 1).permute(0, 3, 1, 2).contiguous()
                var['normal_recon_map_NN_{}'.format(view_id)] = \
                    var['normal_recon_NN_{}'.format(view_id)].view(batch_size, opt.H, opt.W, 3).permute(0, 3, 1, 2).contiguous()

    def compute_loss(self, opt, var, training=False):
        loss = edict()
        batch_size = len(var.idx)
        if opt.loss_weight.render is not None:
            loss.render = self.loss_fns.MSE_loss(var.rgb_recon, var.rgb_gt if 'rgb_gt' in var else var.rgb_input, 
                                                 weight=var.category_weight.view(batch_size, 1, 1) if 'category_weight' in var else None)
        if opt.loss_weight.mask is not None:
            loss.mask = self.loss_fns.mask_loss(var.mask_recon, var.mask_gt if 'mask_gt' in var else var.mask_input, 
                                                weight=var.category_weight.view(batch_size, 1, 1) if 'category_weight' in var else None)
        if opt.loss_weight.normal is not None:
            mask_sup = (var.mask_gt > 0.5) if 'mask_gt' in var else (var.mask_input > 0.5)
            mask_pred = var.mask_recon > 0.5
            loss.normal = self.loss_fns.normal_loss(var.normal_recon, var.normal_transformed, mask_sup & mask_pred, 
                                                    weight=var.category_weight.view(batch_size, 1, 1) if 'category_weight' in var else None, tolerance=opt.reg.normal_tol)
        if opt.loss_weight.eikonal is not None and training:
            loss.eikonal = self.loss_fns.MSE_loss(var.grad_eikonal.view(batch_size, -1), 1, 
                                                  weight=var.category_weight.view(batch_size, 1) if 'category_weight' in var else None)
        if opt.loss_weight.cam_margin is not None and training:
            loss.cam_margin = self.loss_fns.cam_margin_loss(opt, var)
        if opt.loss_weight.cam_uniform is not None and training:
            loss.cam_uniform = self.loss_fns.cam_uniform_loss(opt, var.trig_azim)
        if opt.loss_weight.cam_sym is not None and training:
            loss.cam_sym = self.loss_fns.cam_sym_loss(opt, var, self.estimator)
        if opt.loss_weight.nearest_img is not None and training:
            loss.nearest_img = 0
            for view_id in range(opt.reg.n_views):
                loss.nearest_img += self.loss_fns.MSE_loss(var['rgb_recon_NN_{}'.format(view_id)], var['input_NN_{}'.format(view_id)].rgb_input, 
                                                           weight=var.category_weight.view(batch_size, 1, 1) if 'category_weight' in var else None)
        if opt.loss_weight.nearest_mask is not None and training:    
            loss.nearest_mask = 0
            for view_id in range(opt.reg.n_views):
                loss.nearest_mask += self.loss_fns.mask_loss(var['mask_recon_NN_{}'.format(view_id)], var['input_NN_{}'.format(view_id)].mask_input, 
                                                             weight=var.category_weight.view(batch_size, 1, 1) if 'category_weight' in var else None)
        if opt.loss_weight.nearest_normal is not None and training:
            loss.nearest_normal = 0
            for view_id in range(opt.reg.n_views):
                mask_sup = var['input_NN_{}'.format(view_id)].mask_input > 0.5
                mask_pred = var['mask_recon_NN_{}'.format(view_id)] > 0.5
                loss.nearest_normal += self.loss_fns.normal_loss(
                    var['normal_recon_NN_{}'.format(view_id)], 
                    camera.transform_normal(var['input_NN_{}'.format(view_id)].normal_input, var['pose_NN_{}'.format(view_id)]),
                    mask_sup & mask_pred, 
                    weight=var.category_weight.view(batch_size, 1, 1) if 'category_weight' in var else None,
                    tolerance=opt.reg.normal_tol
                )
        return loss

    def pred_pose(self, opt, var, pred_NN=False, given_input=None):
        device = var.rgb_input_map.device
        img_viewpoint = given_input if given_input is not None else var.rgb_input_map
        trig_azim, trig_elev, trig_theta, scale_focal, scale_dist = self.estimator(img_viewpoint)
        
        # extr
        Ry = camera.azim_to_rotation_matrix(trig_azim, representation='trig')
        Rx = camera.elev_to_rotation_matrix(trig_elev, representation='trig')
        Rz = camera.roll_to_rotation_matrix(trig_theta, representation='trig')
        R_permute = torch.tensor([
            [-1, 0, 0],
            [0, 0, -1],
            [0, -1, 0]
        ]).float().to(Ry.device).unsqueeze(0).expand_as(Ry)
        R = Rz@Rx@Ry@R_permute
        pose_R = camera.pose(R=R)
        trans_z = scale_dist * opt.camera.dist
        trans = torch.stack([torch.zeros_like(trans_z), torch.zeros_like(trans_z), trans_z], dim=-1)
        pose_T = camera.pose(t=trans)
        pose = camera.pose.compose([pose_R, pose_T]).to(device)
        
        # intr
        intr = camera.get_intr(opt, scale_focal)
        
        if not pred_NN:
            var.trig_azim, var.trig_elev, var.trig_theta, var.scale_focal, var.scale_dist = trig_azim, trig_elev, trig_theta, scale_focal, scale_dist
        return pose, intr, scale_dist

    # generate [n_views, 3, 4] pose that rotates by azimuth for visualization
    @torch.no_grad()
    def get_rotate_pose(self, opt, var, n_views=50):
        device = var.rgb_input_map.device
        range_list = opt.data[opt.data.dataset]
        angle_azim = torch.linspace(0, 2, n_views).to(device).view(n_views, 1) * np.pi
        mean_elev = torch.zeros(n_views, 1).to(device) + (range_list.elev_range[1] + range_list.elev_range[0]) / 2 + 15
        mean_theta = torch.zeros(n_views, 1).to(device) + (range_list.theta_range[1] + range_list.theta_range[0]) / 2
        angle_elev = mean_elev * np.pi/180
        angle_theta = mean_theta * np.pi/180
        trig_azim = torch.cat([torch.cos(angle_azim), torch.sin(angle_azim)], dim=1)
        trig_elev = torch.cat([torch.cos(angle_elev), torch.sin(angle_elev)], dim=1)
        trig_theta = torch.cat([torch.cos(angle_theta), torch.sin(angle_theta)], dim=1)
        # get pose
        Ry = camera.azim_to_rotation_matrix(trig_azim, representation='trig')
        Rx = camera.elev_to_rotation_matrix(trig_elev, representation='trig')
        Rz = camera.roll_to_rotation_matrix(trig_theta, representation='trig')
        R_permute = torch.tensor([
            [-1, 0, 0],
            [0, 0, -1],
            [0, -1, 0]
        ]).float().to(Ry.device).unsqueeze(0).expand_as(Ry)
        R = Rz @ Rx @ Ry @ R_permute
        pose = camera.pose(R=R).to(device)
        pose_cam = camera.pose(t=[0, 0, opt.camera.dist]).to(device)
        var.vis_pose = camera.pose.compose([pose, pose_cam]).to(device)
        return var

