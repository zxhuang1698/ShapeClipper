import os, time, shutil
import importlib, tqdm
import numpy as np
import torch
import torch.nn.functional as torch_F
import torch.utils.tensorboard
import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (2048, rlimit[1]))
from copy import deepcopy

import utils.util as util
import utils.util_vis as util_vis
import utils.eval_3D as eval_3D
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.util import log, AverageMeter, toggle_grad, compute_grad2, setup, cleanup
from utils.util import EasyDict as edict

# ============================ main engine for training and evaluation ============================

class Runner():

    def __init__(self, opt):
        super().__init__()
        if os.path.isdir(opt.output_path) and opt.resume == False and opt.device == 0:
            for filename in os.listdir(opt.output_path):
                if "tfevents" in filename: 
                    os.remove(os.path.join(opt.output_path, filename))
                if "vis" in filename: 
                    shutil.rmtree(os.path.join(opt.output_path, filename))
        if opt.device == 0: 
            os.makedirs(opt.output_path,exist_ok=True)
        if opt.world_size > 1: 
            setup(opt.device, opt.world_size, opt.port)
        opt.batch_size = opt.batch_size // opt.world_size
        self.optimizer = getattr(torch.optim, opt.optim.algo)

    # concatenate samples along the batch dimension
    # the input is a list, with each element a dictionary
    # the element of this dictionary is either tensor or dictionary containing tensors
    def concat_samples(self, sample_list):
        stacked = sample_list[0]
        for key, value in stacked.items():
            if isinstance(value, torch.Tensor):
                tensor_list = [value]
                for sample in sample_list[1:]:
                    tensor_list.append(sample[key])
                stacked[key] = torch.cat(tensor_list, dim=0)
            elif isinstance(value, dict):
                for key_sub, value_sub in value.items():
                    assert isinstance(value_sub, torch.Tensor)
                    tensor_list = [value_sub]
                    for sample in sample_list[1:]:
                        tensor_list.append(sample[key][key_sub])
                    stacked[key][key_sub] = torch.cat(tensor_list, dim=0)
            else:
                raise NotImplementedError
        return stacked
    
    def append_viz_data(self, opt):
        # append data for visualization
        cat_samples = [0] * opt.data.num_classes
        viz_data = {}
        viz_data_list = []
        n_vis_classes = opt.eval.n_vis_classes if 'n_vis_classes' in opt.eval else opt.data.num_classes
        if n_vis_classes > opt.data.num_classes: n_vis_classes = opt.data.num_classes
        while sum(cat_samples) < n_vis_classes:
            # load the current batch, a dictionary
            current_batch = next(self.viz_loader_iter)
            # fetch the category of each sample
            for i, cat_tensor in enumerate(current_batch['category_label']):
                cat_idx = cat_tensor.item()
                # check whether we have already got enough samples for this category
                if cat_samples[cat_idx] >= 1: continue
                # if not include the current data
                cat_samples[cat_idx] += 1
                viz_data = {}            
                for key, value in current_batch.items():
                    if isinstance(value, torch.Tensor):
                        viz_data[key] = value[i].unsqueeze(0)
                    elif isinstance(value, dict):
                        viz_data[key] = {}
                        for key_sub, value_sub in value.items():
                            assert isinstance(value_sub, torch.Tensor)
                            viz_data[key][key_sub] = value_sub[i].unsqueeze(0)
                    else:
                        raise NotImplementedError
                viz_data_list.append(viz_data)
        self.viz_data += viz_data_list
    
    def load_dataset(self, opt, eval_split="val"):
        data = importlib.import_module("data.{}".format(opt.data.dataset))
        
        if opt.device == 0: 
            log.info("loading training data...")
        self.train_data = data.Dataset(opt, split="train")
        self.train_loader = self.train_data.setup_loader(opt, shuffle=True)
        self.num_batches = len(self.train_loader)
        
        if opt.device == 0: 
            log.info("loading test data...")
        self.test_data = data.Dataset(opt, split=eval_split)
        self.test_loader = self.test_data.setup_loader(opt, shuffle=False, drop_last=False, batch_size=opt.eval.batch_size)
        
        if opt.device == 0:
            log.info("creating data for visualization...")
            self.viz_loader = self.test_data.setup_loader(opt, shuffle=True, drop_last=False, batch_size=opt.eval.batch_size)
            self.viz_loader_iter = iter(self.viz_loader)
            self.viz_data = []
            for _ in range(opt.eval.n_vis):
                self.append_viz_data(opt)

    def build_networks(self, opt):
        if opt.device == 0: 
            log.info("building networks...")
        graph_name = 'pretrain' if opt.pretrain else 'graph'
        module = importlib.import_module("model.{}".format(graph_name))
        if opt.world_size == 1: 
            self.graph = torch.nn.DataParallel(module.Graph(opt).to(opt.device))
        else: 
            self.graph = DDP(module.Graph(opt).to(opt.device), device_ids=[opt.device], find_unused_parameters=True)

    def setup_optimizer(self, opt):
        if opt.device == 0: 
            log.info("setting up optimizers...")
        kwargs = {}
        for k, v in opt.optim.params.items():
            if k == 'betas': 
                kwargs[k] = tuple(v)
            else: 
                kwargs[k] = v

        optim_list_full = []
        optim_list_V = []
        for k, v in self.graph.named_parameters():
            optim_list_full.append(v)
            if 'estimator' in k: 
                optim_list_V.append(v)
                
        optim_dict_full = [dict(params=optim_list_full, lr=opt.optim.lr), ]
        optim_dict_V = [dict(params=optim_list_V, lr=opt.optim.lr), ]
        self.optim_full = self.optimizer(optim_dict_full, **kwargs)
        self.optim_V = self.optimizer(optim_dict_V, **kwargs)

    def restore_checkpoint(self, opt, best=False, evaluate=False):
        epoch_start, iter_start = None, None
        if opt.resume:
            if opt.device == 0: 
                log.info("resuming from previous checkpoint...")
            epoch_start, iter_start, best_val = util.restore_checkpoint(opt, self, resume=opt.resume, best=best if opt.data.dataset != 'openimage' else False, evaluate=evaluate)
            self.best_val = best_val
        elif opt.load is not None:
            if opt.device == 0: 
                log.info("loading weights from checkpoint {}...".format(opt.load))
            epoch_start, iter_start, best_val = util.restore_checkpoint(opt, self, load_name=opt.load)
        else:
            if opt.device == 0: 
                log.info("initializing weights from scratch...")
        self.epoch_start = epoch_start or 0
        self.iter_start = iter_start or 0

    def setup_visualizer(self, opt):
        if opt.device == 0: 
            log.info("setting up visualizers...")
            if opt.tb:
                self.tb = torch.utils.tensorboard.SummaryWriter(log_dir=opt.output_path, flush_secs=10)
    
    def train(self, opt):
        # before training
        if opt.device == 0: 
            log.title("TRAINING START")
        self.graph.module.estimator.reset_scales()
        self.timer = edict(start=time.time(), it_mean=None)
        self.iter_skip = self.iter_start % len(self.train_loader)
        self.it = self.iter_start
        if not opt.resume: 
            self.best_val = np.inf
            self.best_ep = 1

        # training
        if self.iter_start == 0 and opt.device == 0: 
            self.evaluate(opt, ep=0, training=True)
        for self.ep in range(self.epoch_start, opt.max_epoch):
            self.train_epoch(opt)

        # after training
        if opt.device == 0: 
            self.save_checkpoint(opt, ep=self.ep+1, it=self.it, best_val=self.best_val)
        if opt.tb and opt.device == 0:
            self.tb.flush()
            self.tb.close()
        if opt.device == 0:
            log.title("TRAINING DONE")
            log.info("Best CD: %.4f @ epoch %d" % (self.best_val, self.best_ep))
        if opt.world_size > 1: 
            cleanup()
    
    def train_epoch(self, opt):
        # before train epoch
        if opt.world_size > 1: 
            torch.distributed.barrier() 
            self.train_loader.sampler.set_epoch(self.ep)
        batch_progress = tqdm.tqdm(range(self.num_batches),desc="training epoch {}".format(self.ep+1),leave=False) \
            if opt.device == 0 else range(self.num_batches)
        self.graph.train()
        
        # train epoch
        loader = iter(self.train_loader) 
        for batch_id in batch_progress:
            # if resuming from previous checkpoint, skip until the last iteration number is reached
            if self.iter_skip>0:
                if opt.device == 0: 
                    batch_progress.set_description("(fast-forwarding...)")
                self.iter_skip -= 1
                if self.iter_skip == 0 and opt.device == 0: 
                    batch_progress.set_description("training epoch {}".format(self.ep+1))
                continue
            
            # train iteration
            batch = next(loader)
            var = edict(batch)
            opt.H, opt.W = opt.image_size
            var = util.move_to_device(var, opt.device)
            loss = self.train_iteration(opt, var, batch_progress)
            
        # after train epoch
        if opt.device == 0: log.loss_train(opt, self.ep+1, opt.optim.lr, loss, self.timer)
        if (self.ep + 1) % opt.freq.eval == 0 and opt.device == 0: 
            current_val = self.evaluate(opt, ep=self.ep+1, training=True)
            if current_val < self.best_val:
                self.best_val = current_val
                self.best_ep = self.ep + 1
                self.save_checkpoint(opt, ep=self.ep+1, it=self.it, best_val=self.best_val, best=True, latest=True)

    def train_iteration(self, opt, var, loader):
        # before train iteration
        self.timer.it_start = time.time()
        non_act_loss_key = []
        
        # train iteration
        if self.it > opt.optim.iter_camera:
            optim = self.optim_full
        else:
            for module in self.graph.modules():
                if isinstance(module, torch.nn.BatchNorm2d):
                    module.eval()
            optim = self.optim_V
            non_act_loss_key.append('nearest_img')
            non_act_loss_key.append('nearest_mask')
            non_act_loss_key.append('nearest_normal')
            non_act_loss_key.append('eikonal')
        optim.zero_grad()
        var, loss = self.graph.forward(opt, var, training=True, get_loss=True)
        loss = self.summarize_loss(opt, var, loss, non_act_loss_key=non_act_loss_key)
        loss.all.backward()
        optim.step()
        
        # after train iteration
        if opt.device == 0:
            if (self.it) % opt.freq.vis == 0: 
                self.visualize(opt, var, step=self.it, split="train")
            if (self.it+1) % opt.freq.ckpt_latest == 0: 
                self.save_checkpoint(opt, ep=self.ep, it=self.it+1, best_val=self.best_val, latest=True)
            if (self.it) % opt.freq.scalar == 0: 
                self.log_scalars(opt, var, loss, step=self.it, split="train")
                self.tb.add_scalar("train/beta", self.graph.module.renderer.density.beta, global_step=self.it)
            if (self.it) % opt.freq.save_vis == 0:
                with torch.no_grad():
                    opt.H, opt.W = opt.eval.image_size
                    self.graph.eval()
                    for i in range(len(self.viz_data)):
                        var_viz = edict(deepcopy(self.viz_data[i]))
                        var_viz = util.move_to_device(var_viz, opt.device)
                        var_viz = self.graph.module(opt, var_viz, training=False, visualize=False, get_loss=False)
                        vis_folder = "vis_log/iter_{}".format(self.it)
                        os.makedirs("{}/{}".format(opt.output_path, vis_folder), exist_ok=True)
                        util_vis.dump_images(opt, var_viz.idx, "image_input", var_viz.rgb_input_map, masks=None, from_range=(0, 1), folder=vis_folder)
                        util_vis.dump_images(opt, var_viz.idx, "image_recon", var_viz.rgb_recon_map, masks=var_viz.mask_hard_map, from_range=(0, 1), poses=var_viz.pose, folder=vis_folder)
                        util_vis.dump_images(opt, var_viz.idx, "mask_recon", var_viz.mask_recon_map, folder=vis_folder)
                        util_vis.dump_images(opt, var_viz.idx, "mask_input", var_viz.mask_input_map, folder=vis_folder)
                        if 'normal_input_map' not in var_viz:
                            var_viz.normal_input_map = var_viz.normal_gt.view(len(var_viz.idx), opt.image_size[0], opt.image_size[1], 3).permute(0, 3, 1, 2).contiguous()
                        util_vis.dump_images(opt, var_viz.idx, "normal_input_viewpoint", var_viz.normal_input_map, masks=None, from_range=(-1, 1), folder=vis_folder)
                        util_vis.dump_images(opt, var_viz.idx, "normal_input_canonical", var_viz.normal_transformed_map, masks=None, from_range=(-1, 1), folder=vis_folder)
                        util_vis.dump_images(opt, var_viz.idx, "normal_recon", var_viz.normal_recon_map, masks=None, from_range=(-1, 1), folder=vis_folder)
                    self.graph.train()
        self.it += 1
        if opt.device == 0: 
            loader.set_postfix(it=self.it, loss="{:.3f}".format(loss.all))
        self.timer.it_end = time.time()
        util.update_timer(opt, self.timer, self.ep, len(loader))
        return loss

    def summarize_loss(self, opt, var, loss, non_act_loss_key=[]):
        loss_all = 0.
        assert("all" not in loss)
        # weigh losses
        for key in loss:
            assert(key in opt.loss_weight)
            if opt.loss_weight[key] is not None:
                assert not torch.isinf(loss[key].mean()), "loss {} is Inf".format(key)
                assert not torch.isnan(loss[key].mean()), "loss {} is NaN".format(key)
                loss_all += float(opt.loss_weight[key])*loss[key].mean() if key not in non_act_loss_key else 0.0*loss[key].mean()
        loss.update(all=loss_all)
        return loss

    @torch.no_grad()
    def evaluate(self, opt, ep, training=False):
        self.graph.eval()
        
        # change height and width
        opt.H, opt.W = opt.eval.image_size

        # params for metrics
        f_scores = []
        loss_eval = edict()
        metric_eval = dict(dist_acc=0., dist_cov=0.)
        acc_cat = [0.] * opt.data.num_classes
        comp_cat = [0.] * opt.data.num_classes
        counts_cat = [0.001] * opt.data.num_classes
        
        # dataloader on the test set
        loader = tqdm.tqdm(self.test_loader, desc="evaluating", leave=False)

        for it, batch in enumerate(loader):

            # inference the model
            var = edict(batch)
            var = self.evaluate_batch(opt, var, ep, it, single_gpu=True)

            # record CD for evaluation
            dist_acc, dist_cov = eval_3D.eval_metrics(opt, var, self.graph.module.sdf_network)
            
            # accumulate f-score
            f_scores.append(var.f_score)
        
            # accumulate CD by category
            for i in range(len(var.idx)):
                cat_idx = var.category_label[i].item()
                counts_cat[cat_idx] += 1
                acc_cat[cat_idx] += var.cd_acc[i].item()
                comp_cat[cat_idx] += var.cd_comp[i].item()

            metric_eval["dist_acc"] += dist_acc*len(var.idx)
            metric_eval["dist_cov"] += dist_cov*len(var.idx)
            loader.set_postfix(CD="{:.3f}".format((dist_acc + dist_cov) / 2))

            # save the predicted mesh for vis data if in train mode
            if it == 0 and training: 
                for i in range(len(self.viz_data)):
                    var_viz = edict(deepcopy(self.viz_data[i]))
                    var_viz = self.evaluate_batch(opt, var_viz, ep, it, single_gpu=True, visualize=True)
                    var_viz = self.graph.module.get_rotate_pose(opt, var_viz, n_views=50)
                    self.vis_rotate(opt, var_viz, n_views=50)
                    eval_3D.eval_metrics(opt, var_viz, self.graph.module.sdf_network, vis_only=True)
                    self.visualize(opt, var_viz, step=ep, split="eval")
                    self.dump_results(opt, var_viz, ep, train=True)
            
            # dump the result if in eval mode
            if not training: 
                self.dump_results(opt, var, ep, write_new=(it == 0))

        # save the per-cat evaluation metrics
        if not training:
            per_cat_cd_file = os.path.join(opt.output_path, 'cd_cat.txt')
            with open(per_cat_cd_file, "w") as outfile:
                outfile.write("CD     Acc    Comp   Count Cat\n")
                for i in range(opt.data.num_classes):
                    acc_i = acc_cat[i] / counts_cat[i]
                    comp_i = comp_cat[i] / counts_cat[i]
                    cd_i = (acc_i + comp_i) / 2
                    outfile.write("%.4f %.4f %.4f %5d %s\n" % (cd_i, acc_i, comp_i, counts_cat[i], self.test_data.label2cat[i]))
                    
            # print f_scores
            f_scores = torch.cat(f_scores, dim=0).mean(dim=0)
            print('##############################')
            for i in range(len(opt.eval.f_thresholds)):
                print('F-score @ %.2f: %.4f' % (opt.eval.f_thresholds[i]*100, f_scores[i].item()))
            print('##############################')

            # write to file
            f_score_file = os.path.join(opt.output_path, 'f_score.txt')
            with open(f_score_file, "w") as outfile:
                for i in range(len(opt.eval.f_thresholds)):
                    outfile.write('F-score @ %.2f: %.4f\n' % (opt.eval.f_thresholds[i]*100, f_scores[i].item()))

        # average the metric        
        for key in metric_eval: metric_eval[key] /= len(self.test_data)
        
        # print eval info
        log.loss_eval(opt, loss=None, chamfer=(metric_eval["dist_acc"], metric_eval["dist_cov"]))
        
        # return the cd to decide best checkpoint during training
        val_metric = (metric_eval["dist_acc"] + metric_eval["dist_cov"]) / 2
        torch.cuda.empty_cache()
        return val_metric.item()

    def evaluate_batch(self, opt, var, ep=None, it=None, single_gpu=False, visualize=False):
        var = util.move_to_device(var, opt.device)
        if single_gpu:
            var  = self.graph.module(opt, var, training=False, visualize=visualize, get_loss=False)
        else:
            var = self.graph(opt, var, training=False, visualize=visualize, get_loss=False)
        return var

    def vis_rotate(self, opt, var, n_views=50, vis_NN=False):
        batch_size = len(var.idx)
        imgs_list = []
        masks_list = []
        normals_list = []
        for i in range(n_views):
            pose_i = var.vis_pose[i].unsqueeze(0).expand(batch_size, -1, -1)
            # [B, 3, H, W] and [B, 1, H, W]
            imgs_view_i, masks_view_i, masks_hard_view_i, _, normals_view_i, _ = \
                self.graph.module.renderer(opt, pose_i, var.intr, torch.ones_like(var.scale_dist), var.proj_latent_sdf, 
                                           var.proj_latent_rgb_NN if vis_NN else var.proj_latent_rgb, training=False)
            imgs_view_i = imgs_view_i.view(batch_size, opt.H, opt.W, 3).permute(0, 3, 1, 2).contiguous()
            masks_view_i = masks_view_i.view(batch_size, opt.H, opt.W, 1).permute(0, 3, 1, 2).contiguous()
            masks_hard_view_i = masks_hard_view_i.view(batch_size, opt.H, opt.W, 1).permute(0, 3, 1, 2).contiguous()
            imgs_list.append(imgs_view_i)
            masks_list.append(masks_view_i)
            normals_view_i = normals_view_i.view(batch_size, opt.H, opt.W, 3).permute(0, 3, 1, 2).contiguous()
            normals_view_i = normals_view_i / 2 + 0.5
            normals_list.append(normals_view_i)
        var.rotating_imgs = imgs_list
        var.rotating_masks = masks_list
        var.rotating_normals = normals_list

    @torch.no_grad()
    def log_scalars(self, opt, var, loss, metric=None, step=0, split="train"):
        if split=="train":
            dist_acc, dist_cov = eval_3D.eval_metrics(opt, var, self.graph.module.sdf_network)
            metric = dict(dist_acc=dist_acc, dist_cov=dist_cov)
        for key, value in loss.items():
            if key=="all": continue
            self.tb.add_scalar("{0}/loss_{1}".format(split, key), value.mean(), step)
        if metric is not None:
            for key, value in metric.items():
                self.tb.add_scalar("{0}/{1}".format(split, key), value, step)

    @torch.no_grad()
    def visualize(self, opt, var, step=0, split="train"):
        opt.H, opt.W = opt.eval.image_size
        util_vis.tb_image(opt, self.tb, step, split, "image_input_map", var.rgb_input_map, masks=None, from_range=(0, 1), poses=var.pose_gt)
        util_vis.tb_image(opt, self.tb, step, split, "mask_input_map", var.mask_input_map)
        if 'rgb_recon_map' in var:
            util_vis.tb_image(opt, self.tb, step, split, "image_recon_map", var.rgb_recon_map, masks=None, from_range=(0, 1), poses=var.pose)
            util_vis.tb_image(opt, self.tb, step, split, "mask_recon_map", var.mask_recon_map)
        if 'input_NN_0' in var:
            for view_id in range(opt.reg.n_views):
                util_vis.tb_image(opt, self.tb, step, split, "image_input_map_NN_{}".format(view_id), 
                                    var['input_NN_{}'.format(view_id)].rgb_input_map, masks=None, from_range=(0, 1), poses=var['pose_NN_{}'.format(view_id)])
                util_vis.tb_image(opt, self.tb, step, split, "mask_input_map_NN_{}".format(view_id), 
                                    var['input_NN_{}'.format(view_id)].mask_input_map)
        if 'rgb_recon_map_NN_0' in var and 'mask_recon_map_NN_0' in var:
            for view_id in range(opt.reg.n_views):
                util_vis.tb_image(opt, self.tb, step, split, "image_recon_map_NN_{}".format(view_id), 
                                    var['rgb_recon_map_NN_{}'.format(view_id)], masks=None, from_range=(0, 1), poses=var['pose_NN_{}'.format(view_id)])
                util_vis.tb_image(opt, self.tb, step, split, "mask_recon_map_NN_{}".format(view_id), 
                                    var['mask_recon_map_NN_{}'.format(view_id)])
        if 'normal_input_map' in var:
            util_vis.tb_image(opt, self.tb, step, split, "normal_input_viewpoint_map", var.normal_input_map, masks=None, from_range=(-1, 1))
        if 'normal_transformed_map' in var:
            util_vis.tb_image(opt, self.tb, step, split, "normal_input_canonical_map", var.normal_transformed_map, masks=None, from_range=(-1, 1))
        if 'normal_recon_map' in var:
            util_vis.tb_image(opt, self.tb, step, split, "normal_recon_map", var.normal_recon_map, masks=None, from_range=(-1, 1))

    @torch.no_grad()
    def dump_results(self, opt, var, ep, write_new=False, train=False):
        # create the dir
        current_folder = "dump" if train == False else "vis_{}".format(ep)
        os.makedirs("{}/{}/".format(opt.output_path, current_folder), exist_ok=True)
        util_vis.dump_images(opt, var.idx, "image_input", var.rgb_input_map, masks=None, from_range=(0, 1), poses=var.pose_gt, folder=current_folder)
        util_vis.dump_images(opt, var.idx, "image_recon", var.rgb_recon_map, masks=var.mask_hard_map, from_range=(0, 1), poses=var.pose, folder=current_folder)
        util_vis.dump_images(opt, var.idx, "mask_recon", var.mask_recon_map, folder=current_folder)
        util_vis.dump_images(opt, var.idx, "mask_input", var.mask_input_map, folder=current_folder)
        if 'normal_input_map' in var:
            util_vis.dump_images(opt, var.idx, "normal_input_viewpoint", var.normal_input_map, masks=None, from_range=(-1, 1), folder=current_folder)
        if 'normal_transformed_map' in var:
            util_vis.dump_images(opt, var.idx, "normal_input_canonical", var.normal_transformed_map, masks=None, from_range=(-1, 1), folder=current_folder)
        if 'normal_recon_map' in var:
            util_vis.dump_images(opt, var.idx, "normal_recon", var.normal_recon_map, masks=None, from_range=(-1, 1), folder=current_folder)
        if 'input_NN_0' in var:
            for view_id in range(opt.reg.n_views):
                util_vis.dump_images(opt, var.idx, "image_input_NN_{}".format(view_id), var['input_NN_{}'.format(view_id)].rgb_input_map, 
                                        masks=var['input_NN_{}'.format(view_id)].mask_input_map, from_range=(0, 1), poses=var['pose_NN_{}'.format(view_id)], folder=current_folder)
        if 'rgb_recon_map_NN_0' in var and 'mask_recon_map_NN_0' in var:
            for view_id in range(opt.reg.n_views):
                util_vis.dump_images(opt, var.idx, "image_recon_NN_{}".format(view_id), var['rgb_recon_map_NN_{}'.format(view_id)], 
                                        masks=var['mask_recon_map_NN_{}'.format(view_id)], from_range=(0, 1), poses=var['pose_NN_{}'.format(view_id)], folder=current_folder)
        util_vis.dump_meshes(opt, var.idx, "mesh", var.mesh_pred, folder=current_folder)
        if 'dpc' in var:
            util_vis.dump_pointclouds_compare(opt, var.idx, "pointclouds_comp", var.dpc_pred, var.dpc.points, folder=current_folder)
        if train:
            util_vis.dump_gifs(opt, var.idx, "image_rotate", var.rotating_imgs, from_range=(0, 1), folder=current_folder)
            util_vis.dump_gifs(opt, var.idx, "mask_rotate", var.rotating_masks, folder=current_folder)
            util_vis.dump_gifs(opt, var.idx, "normal_rotate", var.rotating_normals, from_range=(0, 1), folder=current_folder)
        else:
            # write chamfer distance results
            chamfer_fname = "{}/chamfer.txt".format(opt.output_path)
            with open(chamfer_fname, "w" if write_new else "a") as file:
                for i, acc, comp in zip(var.idx, var.cd_acc, var.cd_comp):
                    file.write("{} {:.8f} {:.8f}\n".format(i, acc, comp))

    def save_checkpoint(self, opt, ep=0, it=0, best_val=np.inf, latest=False, best=False):
        assert opt.device == 0
        util.save_checkpoint(opt, self, ep=ep, it=it, best_val=best_val, latest=latest, best=best)
        if not latest:
            log.info("checkpoint saved: ({0}) {1}, epoch {2} (iteration {3})".format(opt.group, opt.name, ep, it))
        if best:
            log.info("Saving the current model as the best...")
