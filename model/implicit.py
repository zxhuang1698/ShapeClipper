import torch
import torch.nn as nn
import torch.nn.functional as torch_F
import numpy as np

# positional encoding, from https://github.com/bmild/nerf.
class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, N_freqs)
        else:
            freq_bands = torch.linspace(2.**0., 2.**max_freq, N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn,
                                 freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)

def get_embedder(posenc_res, input_dims=3):
    embed_kwargs = {
        'include_input': True,
        'input_dims': input_dims,
        'max_freq_log2': posenc_res-1,
        'num_freqs': posenc_res,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    def embed(x, eo=embedder_obj): return eo.embed(x)
    return embed, embedder_obj.out_dim

# implicit function based on https://github.com/lioryariv/volsdf.
class Density(nn.Module):
    def __init__(self, params_init={}):
        super().__init__()
        for p in params_init:
            param = nn.Parameter(torch.tensor(params_init[p]))
            setattr(self, p, param)

    def forward(self, sdf, beta=None):
        return self.density_func(sdf, beta=beta)

class LaplaceDensity(Density):
    def __init__(self, params_init={}, beta_min=0.0001):
        super().__init__(params_init=params_init)
        self.beta_min = torch.tensor(beta_min)

    def density_func(self, sdf, beta=None):
        if beta is None:
            beta = self.get_beta()

        alpha = 1 / beta
        output = torch.zeros_like(sdf)
        mask_pos = (sdf >= 0)
        output[mask_pos] = 0.5 * torch.exp(-sdf[mask_pos] / beta)
        output[~mask_pos] = 1 - 0.5 * torch.exp(sdf[~mask_pos] / beta)
        return alpha * output

    def get_beta(self):
        beta = self.beta.abs() + self.beta_min
        return beta

class SDFNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.force_symmetry = opt.arch.force_symmetry
        self.proj_latent_dim = opt.arch.impl_sdf.proj_latent_dim
        self.n_hidden = opt.arch.impl_sdf.n_hidden_layers
        self.n_channel = opt.arch.impl_sdf.n_channels
        
        dims = [3 + self.proj_latent_dim] + [self.n_channel] * self.n_hidden + [1 + self.n_channel]
        self.num_layers = len(dims)
        self.skip_in = opt.arch.impl_sdf.skip_connection

        # define positional embedder
        posenc_res = opt.arch.impl_sdf.pos_enc
        self.embed_fn = None
        if posenc_res > 0:
            embed_fn, input_ch = get_embedder(posenc_res, input_dims=3)
            self.embed_fn = embed_fn
            dims[0] += (input_ch - 3)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            if l in self.skip_in:
                in_dim = dims[l] + dims[0]
            else:
                in_dim = dims[l]

            # define and initialize linear layer
            lin = nn.Linear(in_dim, out_dim)
            if opt.arch.impl_sdf.geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(in_dim), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -opt.arch.impl_sdf.init_sphere_radius)
                elif posenc_res > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif posenc_res > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            # reparameterize if needed
            if opt.arch.impl_sdf.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)

        # activation
        self.softplus = nn.Softplus(beta=100)

    def forward(self, points_raw, proj_latent):
        if self.force_symmetry:
            # make first dimension of input coordinates to be positive
            # so that symmetry on yz plane is forced
            points = points_raw.clone()
            points[..., 0] = torch.abs(points[..., 0].clone())
        else:
            points = points_raw
        
        # positional encoding
        if self.embed_fn is not None:
            points = self.embed_fn(points)

        # forward by layer
        inputs = torch.cat([points, proj_latent], dim=-1)
        x = inputs
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            if l in self.skip_in:
                x = torch.cat([x, inputs], 1) / np.sqrt(2)
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.softplus(x)
        return x

    def get_conditional_output(self, opt, batch_size, points_flat, proj_latent, compute_grad=True):
        # prepare the latent code for MLP input
        N = points_flat.shape[0] // batch_size
        proj_latent = proj_latent.unsqueeze(1).repeat(1, N, 1).view(batch_size * N, -1)
        assert proj_latent.shape[1] == opt.arch.impl_sdf.proj_latent_dim
        if compute_grad:
            proj_latent = proj_latent.detach()
        
        # forward the MLP
        points_flat.requires_grad_(True)
        output = self.forward(points_flat, proj_latent)
        sdf = output[:,:1]
        impl_feat = output[:, 1:]
        
        # get the gradient
        if compute_grad:
            d_output = torch.ones_like(sdf, requires_grad=False, device=sdf.device)
            gradients = torch.autograd.grad(
                outputs=sdf,
                inputs=points_flat,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
        else:
            gradients = None
        return sdf, impl_feat, gradients

class RGBNetwork(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.force_symmetry = opt.arch.force_symmetry
        self.proj_latent_dim = opt.arch.impl_rgb.proj_latent_dim
        self.n_hidden = opt.arch.impl_rgb.n_hidden_layers
        self.n_sdf_channel = opt.arch.impl_sdf.n_channels
        self.n_channel = opt.arch.impl_rgb.n_channels
        dims = [3 + self.proj_latent_dim + self.n_sdf_channel] + [self.n_channel] * self.n_hidden + [3]
        self.num_layers = len(dims)

        # define positional embedder
        posenc_res = opt.arch.impl_rgb.pos_enc
        self.embed_fn = None
        if posenc_res > 0:
            embed_fn, input_ch = get_embedder(posenc_res, input_dims=3)
            self.embed_fn = embed_fn
            dims[0] += (input_ch - 3)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)
            if opt.arch.impl_rgb.weight_norm:
                lin = nn.utils.weight_norm(lin)
            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, points_raw, proj_latent, sdf_feature):
        if self.force_symmetry:
            # make first dimension of input coordinates to be positive
            # so that symmetry on yz plane is forced
            points = points_raw.clone()
            points[..., 0] = torch.abs(points[..., 0].clone())
        else:
            points = points_raw
        
        if self.embed_fn is not None:
            points = self.embed_fn(points)
        
        x = torch.cat([points, proj_latent, sdf_feature], dim=-1)
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))
            x = lin(x)
            if l < self.num_layers - 2:
                x = self.relu(x)
        x = self.sigmoid(x)
        return x