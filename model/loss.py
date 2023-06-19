import utils.camera as camera
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as torch_F

from copy import deepcopy

class Loss(nn.Module):

    def __init__(self, opt):
        super().__init__()
        self.opt = deepcopy(opt)

    def L1_loss(self, pred, label=0, weight=None):
        loss = (pred.contiguous()-label).abs()
        return self.aggregate_loss(loss, weight=weight)

    def MSE_loss(self, pred, label=0, weight=None, tolerance=0.):
        loss = (pred.contiguous()-label)**2
        if tolerance > 1.e-5:
            assert len(pred.shape) == 3 and pred.shape[2] in [1, 3]
            if pred.shape[2] == 3:
                loss_pixel = loss.mean(dim=2).view(-1)
            else:
                loss_pixel = loss.view(-1)
            loss_sorted = torch.sort(loss_pixel, dim=0, descending=False)[0]
            end_idx = int((1-tolerance) * loss_sorted.shape[0])
            loss_valid = loss_sorted[:end_idx].contiguous()
            assert weight is None
            return self.aggregate_loss(loss_valid, weight=weight)
        return self.aggregate_loss(loss, weight=weight)

    def CE_loss(self, pred, label, weight=None, mask=None):
        loss = torch_F.cross_entropy(pred, label, reduction="none")
        return self.aggregate_loss(loss, weight=weight)

    def BCE_loss(self, pred, label, weight=None, mask=None, tolerance=0.):
        batch_size = pred.shape[0]
        label = label.expand_as(pred)
        loss = torch_F.binary_cross_entropy(pred, label, reduction="none")
        if tolerance > 1.e-5:
            assert len(pred.shape) == 4 and pred.shape[1] == 1
            # [B, HW]
            loss_pixel = loss.view(batch_size, -1)
            loss_sorted = torch.sort(loss_pixel, dim=-1, descending=False)[0]
            end_idx = int((1-tolerance) * loss_pixel.shape[1])
            loss_valid = loss_sorted[:, :end_idx].contiguous()
            return self.aggregate_loss(loss_valid, weight=weight, mask=mask)
        return self.aggregate_loss(loss, weight=weight, mask=mask)

    def normal_loss(self, normal_pred, normal_gt, mask, weight=None, tolerance=0.):
        mask = mask.squeeze(-1)
        assert normal_pred.shape == normal_gt.shape
        # [B, n_rays, 3]
        assert len(normal_pred.shape) == 3
        assert len(mask.shape) == 2
        cos_sim = torch.sum(normal_pred[mask] * normal_gt[mask], dim=-1)
        angular_diff = 1 - cos_sim
        L1_diff = (normal_pred[mask] - normal_gt[mask]).abs().sum(dim=-1)
        loss = self.opt.reg.normal_l1 * L1_diff + angular_diff
        idx_robust = torch.sort(angular_diff, dim=0, descending=False)[1][:int(loss.shape[0] * (1 - tolerance))]
        if weight is not None: 
            weight = weight.expand_as(normal_pred)[mask][..., 0]
            loss = loss * weight
        loss_robust = loss[idx_robust].mean()
        return loss_robust

    def aggregate_loss(self, loss, weight=None):
        if weight is not None:
            loss = loss * weight
        loss = loss.mean()
        return loss
            
    def iou_loss(self, inputs, targets, weight=None, tolerance=0.):
        batch_size = inputs.shape[0]
        inputs_expand = inputs.view(batch_size, -1).contiguous()
        targets_expand = targets.view(batch_size, -1).contiguous()
        if tolerance > 1.e-5:
            assert weight is None
            n_pixels = inputs_expand.shape[1]
            diff = (inputs_expand-targets_expand).abs().view(batch_size * n_pixels)
            idx_sorted = torch.sort(diff, dim=0, descending=False)[1]
            end_idx = int((1-tolerance) * diff.shape[0])
            idx_outlier = idx_sorted[end_idx:]
            inputs_expand.view(batch_size * n_pixels)[idx_outlier] = targets_expand.view(batch_size * n_pixels)[idx_outlier]
        loss = 1 - (inputs_expand * targets_expand).sum(dim=1) / (inputs_expand + targets_expand - inputs_expand * targets_expand + 1.e-8).sum(dim=1)
        if weight is not None:
            loss = loss * weight.squeeze(1).squeeze(1)
        loss = loss.mean()
        return loss

    def mask_loss(self, inputs, targets, weight=None, tolerance=0.):
        iou_loss = self.iou_loss(inputs, targets, weight=weight, tolerance=tolerance)
        mse_loss = self.MSE_loss(inputs, targets, weight=weight, tolerance=tolerance)
        loss = iou_loss + self.opt.reg.mask_mse * mse_loss
        return loss

    def cam_margin(self, opt, trig, ranges, eps=5):
        assert ranges[0] > -180 and ranges[1] < 180
        cos = trig[:, 0]
        sin = trig[:, 1]
        angle = torch.atan2(sin, cos) * 180/np.pi
        loss = self.L1_loss((-angle+ranges[0]-eps).relu_()) + self.L1_loss((angle-ranges[1]-eps).relu_())
        return loss

    def cam_margin_loss(self, opt, var):
        range_list = opt.data[opt.data.dataset]
        loss = self.cam_margin(opt, var.trig_elev, range_list.elev_range)
        loss += self.cam_margin(opt, var.trig_theta, range_list.theta_range)
        return loss
    
    def cam_sym_loss(self, opt, var, estimator):
        trig_azim_flipped, trig_elev_flipped, trig_theta_flipped, _, _ = estimator(var.rgb_input_map.flip(dims=[3]))
        
        # azim_flipped = -azim (sin flipped, cos unchanged)
        cos_azim_flipped, sin_azim_flipped = trig_azim_flipped[:, 0], trig_azim_flipped[:, 1]
        cos_azim_sup, sin_azim_sup = var.trig_azim[:, 0], -var.trig_azim[:, 1]
        loss_azim = (cos_azim_sup - cos_azim_flipped)**2 + (sin_azim_sup - sin_azim_flipped)**2
        
        # elev_flipped = elev (sin unchanged, cos unchanged)
        cos_elev_flipped, sin_elev_flipped = trig_elev_flipped[:, 0], trig_elev_flipped[:, 1]
        cos_elev_sup, sin_elev_sup = var.trig_elev[:, 0], var.trig_elev[:, 1]
        loss_elev = (cos_elev_sup - cos_elev_flipped)**2 + (sin_elev_sup - sin_elev_flipped)**2
        
        # theta_flipped = -theta (sin flipped, cos unchanged)
        cos_theta_flipped, sin_theta_flipped = trig_theta_flipped[:, 0], trig_theta_flipped[:, 1]
        cos_theta_sup, sin_theta_sup = var.trig_theta[:, 0], -var.trig_theta[:, 1]
        loss_theta = (cos_theta_sup - cos_theta_flipped)**2 + (sin_theta_sup - sin_theta_flipped)**2
        
        loss = loss_azim.mean() + loss_elev.mean() + loss_theta.mean()
        return loss

    def cam_uniform_loss(self, opt, trig):
        # get the empirical distribution
        batch_size = trig.shape[0]
        cos_empr = trig[:, 0]
        sin_empr = trig[:, 1]
        prod_empr = cos_empr * sin_empr

        # [0, 2pi], get the prior
        grid_points = torch.arange(1., 2*batch_size, 2., requires_grad=False).float().to(trig.device) * np.pi / batch_size
        cos_prior = torch.cos(grid_points)
        sin_prior = torch.sin(grid_points)
        prod_prior = cos_prior * sin_prior

        # wasserstein dist
        # sort the empr and prior
        cos_empr = cos_empr.sort(dim=0, descending=False)[0]
        sin_empr = sin_empr.sort(dim=0, descending=False)[0]
        prod_empr = prod_empr.sort(dim=0, descending=False)[0]
        cos_prior = cos_prior.sort(dim=0, descending=False)[0]
        sin_prior = sin_prior.sort(dim=0, descending=False)[0]
        prod_prior = prod_prior.sort(dim=0, descending=False)[0]

        # get the dist
        cos_dists = cos_prior - cos_empr
        sin_dists = sin_prior - sin_empr
        prod_dists = prod_prior - prod_empr

        # wasserstein distance
        if opt.reg.emd_p == 1:
            loss = (cos_dists.abs().mean() + sin_dists.abs().mean() + prod_dists.abs().mean()) / 3
        else:
            loss = (torch.norm(cos_dists, dim=0, p=opt.reg.emd_p) + torch.norm(sin_dists, dim=0, p=opt.reg.emd_p) \
                    + torch.norm(prod_dists, dim=0, p=opt.reg.emd_p)) / (3*batch_size)
        return loss

    def category_reg_loss(self, opt, var, shape_center):
        shape_code_normed = torch_F.normalize(var.proj_latent_sdf, dim=-1)
        shape_center_normed = torch_F.normalize(shape_center, dim=-1)
        logits = shape_code_normed @ shape_center_normed.permute(1, 0).contiguous()
        loss = self.CE_loss(logits/0.3, var.category_label)
        return loss