import numpy as np
import os, time
import torch
import tqdm
from utils.util import EasyDict as edict

from . import graph
import utils.util as util
import importlib
from utils.util import log
import shutil

# ============================ main engine for training and evaluation ============================

class Runner():

    def __init__(self, opt):
        super().__init__()
        if os.path.isdir(opt.output_path) and opt.resume == False:
            for filename in os.listdir(opt.output_path):
                if "tfevents" in filename: 
                    os.remove(os.path.join(opt.output_path, filename))
                if "vis" in filename: 
                    shutil.rmtree(os.path.join(opt.output_path, filename))
        os.makedirs(opt.output_path, exist_ok=True)
        self.optimizer = getattr(torch.optim, opt.optim.algo)

    def load_dataset(self, opt, eval_split="train"):
        data = importlib.import_module("data.{}".format(opt.data.dataset))
        log.info("loading pretrain data...")
        self.pretrain_data = data.Dataset(opt, split="train")
        self.pretrain_loader = self.pretrain_data.setup_loader(opt, shuffle=True, batch_size=opt.batch_size, allow_ddp=False)

    def build_networks(self, opt):
        log.info("building networks...")
        self.graph = torch.nn.DataParallel(Graph(opt).to(opt.device))

    def setup_optimizer(self, opt):
        log.info("setting up optimizers...")
        optim_kwargs = {}
        for k, v in opt.optim.params.items():
            if k == 'betas': 
                optim_kwargs[k] = tuple(v)
            else: 
                optim_kwargs[k] = v

        optim_list = []
        for k, v in self.graph.named_parameters():
            if 'estimator' in k and not 'fc' in k: 
                continue
            optim_list.append(v)
        optim_dict = [dict(params=optim_list, lr=opt.optim.lr), ]
        self.optim = self.optimizer(optim_dict, **optim_kwargs)

    def train(self, opt):
        # before training
        log.title("TRAINING START")
        self.timer = edict(start=time.time(), it_mean=None)
        self.ep = 0
        self.it = 0
        # training
        self.graph.train()
        self.save_checkpoint(opt, ep=self.ep, it=self.it+1, latest=True)
        loader = iter(self.pretrain_loader)
        batch_progress = tqdm.trange(opt.pre.iter, desc="pretraining", leave=False)
        for _ in batch_progress:
            # train iteration
            try:
                batch = next(loader) 
            except StopIteration:
                loader = iter(self.pretrain_loader)
                batch = next(loader) 
            var = edict(batch)
            self.train_iteration(opt, var, batch_progress)
        # after train epoch
        self.save_checkpoint(opt, ep=1, it=self.it)
        # after training
        log.title("TRAINING DONE")

    def train_iteration(self, opt, var, loader):
        # before train iteration
        self.timer.it_start = time.time()
        # train iteration
        self.optim.zero_grad()
        var, loss = self.graph.forward(opt, var)
        loss.all.backward()
        self.optim.step()
        # after train iteration
        if (self.it)%opt.freq.vis==0: 
            self.visualize(opt, var, step=self.it, split="train")
        if (self.it+1)%opt.freq.ckpt_latest==0: 
            self.save_checkpoint(opt, ep=self.ep, it=self.it+1, latest=True)
        self.it += 1
        loader.set_postfix(it=self.it, loss="{:.3f}".format(loss.all))
        self.timer.it_end = time.time()
        util.update_timer(opt, self.timer, self.ep, len(loader))
        return loss
    
    @torch.no_grad()
    def evaluate(self, opt, ep=None, training=False): 
        return

    @torch.no_grad()
    def visualize(self, opt, var, step=0, split="train"): 
        return

    def save_checkpoint(self, opt, ep=0, it=0, latest=False):
        if not opt.pre.viewpoint:
            util.save_checkpoint(opt, self, ep=ep, it=it, best_val=np.inf, children=("reconstructor", "sdf_network", "latent_proj_shape"))
        else:
            util.save_checkpoint(opt, self, ep=ep, it=it, best_val=np.inf, children=("reconstructor", "sdf_network", "latent_proj_shape", "estimator"))
        if not latest:
            log.info("checkpoint saved: ({0}) {1}, epoch {2} (iteration {3})".format(opt.group, opt.name, ep, it))

# ============================ computation graph for forward/backprop ============================

class Graph(graph.Graph):

    def __init__(self, opt):
        super().__init__(opt)

    def forward(self, opt, var):
        var.latent_raw = torch.randn(opt.batch_size, opt.arch.latent_dim_shape+opt.arch.latent_dim_rgb, device=opt.device) * opt.pre.latent_std
        var.latent_shape = var.latent_raw[:, :opt.arch.latent_dim_shape]
        var.latent_rgb = var.latent_raw[:, opt.arch.latent_dim_shape:]

        # get the feature for impl conditioning
        var.proj_latent_sdf = self.latent_proj_shape(var.latent_shape)

        if opt.pre.viewpoint:
            # get the empirical distribution
            trig_azim = self.estimator(var.rgb_input_map)[0]
            batch_size = trig_azim.shape[0]
            cos_azim = trig_azim[:, 0]
            sin_azim = trig_azim[:, 1]
            prod_azim = cos_azim * sin_azim

            # [0, 2pi], get the prior
            grid_points = torch.arange(1., 2*batch_size, 2., requires_grad=False).float().to(trig_azim.device) * np.pi / batch_size
            cos_prior = torch.cos(grid_points)
            sin_prior = torch.sin(grid_points)
            prod_prior = cos_prior * sin_prior

            # wasserstein dist
            # sort the empr and prior
            cos_empr = cos_azim.sort(dim=0, descending=False)[0]
            sin_empr = sin_azim.sort(dim=0, descending=False)[0]
            prod_empr = prod_azim.sort(dim=0, descending=False)[0]
            cos_prior = cos_prior.sort(dim=0, descending=False)[0]
            sin_prior = sin_prior.sort(dim=0, descending=False)[0]
            prod_prior = prod_prior.sort(dim=0, descending=False)[0]
            # get the dist
            cos_dists = cos_prior - cos_empr
            sin_dists = sin_prior - sin_empr
            prod_dists = prod_prior - prod_empr

            # calculate the loss
            var.w_dist = (cos_dists.abs().mean() + sin_dists.abs().mean() + prod_dists.abs().mean()) / 3

        loss = self.compute_loss(opt, var)
        return var, loss

    def compute_loss(self, opt, var):
        loss = edict()
        loss.all = 0
        # density loss
        if opt.pre.density:
            loss.all += self.get_sdf_loss(opt, opt.device, var.proj_latent_sdf)
        # viewpoint loss
        if opt.pre.viewpoint:
            loss.all += var.w_dist
        return loss

    def get_sdf_loss(self, opt, device, proj_latent_sdf):
        lower, upper = opt.pre.sample_range
        points_3D = torch.rand(opt.batch_size * opt.pre.sample_points, 3, device=device)
        points_3D = points_3D * (upper - lower) + lower
        sdf = self.sdf_network.get_conditional_output(opt, opt.batch_size, points_3D, proj_latent_sdf, compute_grad=False)[0]
        sdf_gt = points_3D.norm(dim=-1, keepdim=True) - opt.pre.radius
        loss = self.loss_fns.MSE_loss(sdf, sdf_gt)
        return loss