import torch
import torch.nn as nn
import torch.nn.functional as torch_F
import utils.camera as camera

from model.implicit import LaplaceDensity

class UniformSampler(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.N_samples = opt.render.n_samples_uniform

    def get_z_vals(self, opt, ray_dirs, scale_dist, training=True):
        batch_size = scale_dist.shape[0]
        n_rays = ray_dirs.shape[0] // batch_size
        near = opt.camera.dist * scale_dist.unsqueeze(-1).repeat(1, n_rays).view(ray_dirs.shape[0], 1) - 0.7
        far = opt.camera.dist * scale_dist.unsqueeze(-1).repeat(1, n_rays).view(ray_dirs.shape[0], 1) + 0.7
        # [N_samples, ]
        t_vals = torch.linspace(0., 1., steps=self.N_samples).to(ray_dirs.device)
        # [ray_dirs.shape[0], N_samples]
        z_vals = near * (1. - t_vals) + far * (t_vals)

        if training:
            # get intervals between samples
            mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
            upper = torch.cat([mids, z_vals[..., -1:]], -1)
            lower = torch.cat([z_vals[..., :1], mids], -1)
            # stratified samples in those intervals
            t_rand = torch.rand(z_vals.shape).to(ray_dirs.device)
            z_vals = lower + (upper - lower) * t_rand
        
        # [ray_dirs.shape[0], ]
        idx = torch.randint(z_vals.shape[-1], (z_vals.shape[0],)).to(ray_dirs.device)
        # [ray_dirs.shape[0], 1]
        z_samples_eik = torch.gather(z_vals, 1, idx.unsqueeze(-1))

        return z_vals, z_samples_eik

# renderer based on https://github.com/lioryariv/volsdf.
class Renderer(nn.Module):
    def __init__(self, opt, sdf_network, rgb_network):
        super().__init__()
        self.bg_color = opt.data.bgcolor
        self.eik_range = opt.arch.impl_sdf.eikonal_sample_range
        self.normal_model = opt.render.normal_model

        self.sdf_network = sdf_network
        self.rgb_network = rgb_network
        self.density = LaplaceDensity(params_init={'beta': opt.arch.impl_sdf.beta_init})

        if opt.render.sampler == 'uniform':
            self.ray_sampler = UniformSampler(opt)
            self.N_samples = opt.render.n_samples_uniform
        else:
            raise NotImplementedError

    def forward(self, opt, pose, intr, scale_dist, proj_latent_sdf, proj_latent_rgb, ray_idx=None, training=True, visualize=False):
        # [B, HW, 3]
        cam_loc, ray_dirs_raw = camera.get_center_and_ray(opt, pose, intr=intr, device=pose.device)
        ray_dirs = torch_F.normalize(ray_dirs_raw, dim=-1)
        # factor that converts the ray length (z_val) to real depth: depth = z_val * depth_fac
        depth_fac = ray_dirs.norm(dim=-1, keepdim=True) / ray_dirs_raw.norm(dim=-1, keepdim=True)
        if ray_idx is not None:
            gather_idx = ray_idx[..., None].repeat(1, 1, 3)
            ray_dirs = ray_dirs.gather(dim=1, index=gather_idx)
            depth_fac = depth_fac.gather(dim=1, index=ray_idx[..., None])
            if opt.camera.model == "orthographic":
                cam_loc = cam_loc.gather(dim=1, index=gather_idx)
        batch_size, num_rays, _ = ray_dirs.shape
        
        if opt.camera.model=="perspective":
            cam_loc = cam_loc.repeat(1, num_rays, 1)
        # [B * num_rays, 3]
        cam_loc = cam_loc.reshape(-1, 3)
        ray_dirs = ray_dirs.reshape(-1, 3)
        # [B * num_rays, 1]
        depth_fac = depth_fac.reshape(-1, 1)

        # get the depth, [B * num_rays, N_samples]
        z_vals, z_samples_eik = self.ray_sampler.get_z_vals(opt, ray_dirs, scale_dist, training)
        assert self.N_samples == z_vals.shape[1]

        # get the point coordinates, [B * num_rays, N_samples, 3]
        points = cam_loc.unsqueeze(1) + z_vals.unsqueeze(2) * ray_dirs.unsqueeze(1)
        # [B * num_rays * N_samples, 3]
        points_flat = points.reshape(-1, 3)

        # repeat the proj_latent_rgb [B, C] -> [B * num_rays * N_samples, C]
        proj_latent_rgb_rep = proj_latent_rgb.unsqueeze(1).repeat(1, num_rays * self.N_samples, 1).view(batch_size * num_rays * self.N_samples, -1)
        assert proj_latent_rgb_rep.shape[1] == opt.arch.impl_rgb.proj_latent_dim

        # forward the sdf and rgb network
        if self.normal_model == 'volume':
            with torch.enable_grad():
                points_flat.requires_grad_(True)
                sdf, sdf_feature, _ = self.sdf_network.get_conditional_output(
                    opt, batch_size, points_flat, proj_latent_sdf, compute_grad=False
                )
                density = self.density(sdf)
                d_output = torch.ones_like(density, requires_grad=False, device=sdf.device)
                normal_flat = -torch.autograd.grad(
                    outputs=density,
                    inputs=points_flat,
                    grad_outputs=d_output,
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True)[0]
        else:
            raise NotImplementedError
        rgb_flat = self.rgb_network(points_flat, proj_latent_rgb_rep, sdf_feature)
        # [B * num_rays, N_samples, 3]
        rgb = rgb_flat.reshape(-1, self.N_samples, 3)

        # volume rendering based on the queried values, [B * num_rays, N_samples]
        weights, alphas = self.volume_rendering(z_vals, sdf)
        
        # get depth
        # [B * num_rays, N_samples]
        depth_samples = z_vals * depth_fac
        # [B * num_rays, 1]
        depth_values = torch.sum(weights * depth_samples, 1).unsqueeze(-1)
        depth_output = depth_values.view(batch_size, -1, 1)
        
        # get normal
        if self.normal_model == 'volume':
            # [B * num_rays, N_samples, 3]
            normal = torch_F.normalize(normal_flat, dim=-1, p=2).reshape(-1, self.N_samples, 3)
            normal_weights = weights.unsqueeze(-1) ** opt.reg.normal_pow
            # normal_weights = torch_F.normalize(normal_weights, dim=-2)
            # [B * num_rays, 3]
            normal_values = torch.sum(normal_weights * normal, 1)
            normal_values = torch_F.normalize(normal_values, dim=-1, p=2)
            normal_output = normal_values.view(batch_size, -1, 3)#  if training else None
        elif self.normal_model == 'surface':
            # [B * num_rays, 3]
            normal_output = torch_F.normalize(normal_flat, dim=-1, p=2).view(batch_size, -1, 3)
        else:
            raise NotImplementedError

        # add background
        # [B * num_rays, ]
        acc_map = torch.sum(weights, -1)
        # [B * num_rays, 3]
        rgb_values = torch.sum(weights.unsqueeze(-1) * rgb, 1)
        rgb_output = rgb_values + (1. - acc_map.unsqueeze(1).repeat(1, 3)) * self.bg_color
        mask_output = acc_map
        mask_hard_output = (mask_output > 0.5).float()

        # reshape
        rgb_output = rgb_output.view(batch_size, -1, 3)
        mask_output = mask_output.view(batch_size, -1, 1)
        mask_hard_output = mask_hard_output.view(batch_size, -1, 1)

        if training:
            # Sample points for the eikonal loss
            n_eik_points = batch_size * num_rays
            # [B, num_rays, 3]
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(self.eik_range[0], self.eik_range[1]).to(rgb_output.device).reshape(batch_size, num_rays, 3)

            # add some of the near surface points
            # [B, num_rays, 3]
            eik_near_points = (cam_loc.unsqueeze(1) + z_samples_eik.unsqueeze(2) * ray_dirs.unsqueeze(1)).reshape(batch_size, num_rays, 3)
            # [B * 2 * num_rays, 3]
            eikonal_points = torch.cat([eikonal_points, eik_near_points], 1).reshape(-1, 3)

            # get the gradient of the sampled points for eikonal loss
            _, _, grad_eikonal = self.sdf_network.get_conditional_output(
                    opt, batch_size, eikonal_points, proj_latent_sdf, compute_grad=True
                )
            grad_eikonal = grad_eikonal.norm(2, dim=1)
        else:
            grad_eikonal = None

        if visualize:
            opacity = alphas.reshape(batch_size, -1, 1)
            points_visualize = points_flat.reshape(batch_size, -1, 3)
            transparency_visualize = torch.cat([opacity, 1 - opacity, torch.zeros_like(opacity)], dim=-1)
            rgb_visualize = torch.cat([rgb_flat.reshape(batch_size, -1, 3), opacity], dim=-1)
            idx = torch.randperm(num_rays)[:200].to(opacity.device)
            points_sampled = self.sample_rays_visualize(idx, points_visualize.reshape(batch_size, num_rays, self.N_samples, -1))
            transparency_sampled = self.sample_rays_visualize(idx, transparency_visualize.reshape(batch_size, num_rays, self.N_samples, -1))
            rgb_sampled = self.sample_rays_visualize(idx, rgb_visualize.reshape(batch_size, num_rays, self.N_samples, -1))
            return rgb_output, mask_output, mask_hard_output, depth_output, normal_output, grad_eikonal, points_sampled, transparency_sampled, rgb_sampled
        else:
            return rgb_output, mask_output, mask_hard_output, depth_output, normal_output, grad_eikonal

    def volume_rendering(self, z_vals, sdf):
         # [B * num_rays * N_samples, 1]
        density_flat = self.density(sdf)
        # [B * num_rays, N_samples]
        density = density_flat.reshape(-1, z_vals.shape[1])  

        # [B * num_rays, N_samples - 1]
        dists = z_vals[:, 1:] - z_vals[:, :-1]
        # distance between samples, the last distance is infinity, [B * num_rays, N_samples]
        dists = torch.cat([dists, torch.zeros(dists.shape[0], 1).to(z_vals.device)], -1)

        # LOG SPACE
        # [B * num_rays, N_samples]
        free_energy = dists * density
        shifted_free_energy = torch.cat([torch.zeros(dists.shape[0], 1).to(z_vals.device), free_energy[:, :-1]], dim=-1)  # shift one step
        # maps free_energy to [0, 1], opacity for each point (probability of it is not empty here)
        alpha = 1 - torch.exp(-free_energy)
        # probability of everything is empty up to now, [B * num_rays, N_samples]
        transmittance = torch.exp(-torch.cumsum(shifted_free_energy, dim=-1)) 

        # probability of the ray hits something here
        weights = alpha * transmittance 
        return weights, alpha
        
    # visualization_item: [B, num_rays, num_samples, C] -> [B, sample_rays, C]
    @torch.no_grad()
    def sample_rays_visualize(self, idx, visualization_item):
        visualization_item = visualization_item[:, idx].clone()
        return visualization_item.reshape(visualization_item.shape[0], -1, visualization_item.shape[-1])