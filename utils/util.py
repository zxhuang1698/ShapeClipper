import numpy as np
import os, sys, time
import shutil
import datetime
import torch
import torch.nn.functional as torch_F
import termcolor
import socket
import contextlib
import vigra
import socket
import torch.distributed as dist

# convert to colored strings
def red(message, **kwargs): return termcolor.colored(str(message), color="red", attrs=[k for k, v in kwargs.items() if v is True])
def green(message, **kwargs): return termcolor.colored(str(message), color="green", attrs=[k for k, v in kwargs.items() if v is True])
def blue(message, **kwargs): return termcolor.colored(str(message), color="blue", attrs=[k for k, v in kwargs.items() if v is True])
def cyan(message, **kwargs): return termcolor.colored(str(message), color="cyan", attrs=[k for k, v in kwargs.items() if v is True])
def yellow(message, **kwargs): return termcolor.colored(str(message), color="yellow", attrs=[k for k, v in kwargs.items() if v is True])
def magenta(message, **kwargs): return termcolor.colored(str(message), color="magenta", attrs=[k for k, v in kwargs.items() if v is True])
def grey(message, **kwargs): return termcolor.colored(str(message), color="grey", attrs=[k for k, v in kwargs.items() if v is True])

def get_time(sec):
    d = int(sec//(24*60*60))
    h = int(sec//(60*60)%24)
    m = int((sec//60)%60)
    s = int(sec%60)
    return d, h, m, s

# which color to use for different log info
class Log:
    def __init__(self): pass
    def process(self, pid):
        print(grey("Process ID: {}".format(pid), bold=True))
    def title(self, message):
        print(yellow(message, bold=True, underline=True))
    def info(self, message):
        print(magenta(message, bold=True))
    def options(self, opt, level=0):
        for key, value in sorted(opt.items()):
            if isinstance(value, (dict, EasyDict)):
                print("   "*level+cyan("* ")+green(key)+":")
                self.options(value, level+1)
            else:
                print("   "*level+cyan("* ")+green(key)+":", yellow(value))
    def loss_train(self, opt, ep, lr, loss, timer):
        message = grey("[train] ", bold=True)
        message += "epoch {}/{}".format(cyan(ep, bold=True), opt.max_epoch)
        message += ", lr:{}".format(yellow("{:.2e}".format(lr), bold=True))
        message += ", loss:{}".format(red("{:.3e}".format(loss.all), bold=True))
        message += ", time:{}".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.elapsed)), bold=True))
        message += " (ETA:{})".format(blue("{0}-{1:02d}:{2:02d}:{3:02d}".format(*get_time(timer.arrival))))
        print(message)
    def loss_eval(self, opt, loss=None, chamfer=None):
        message = grey("[eval] ", bold=True)
        if loss is not None: message += "loss:{}".format(red("{:.3e}".format(loss.all), bold=True))
        if chamfer is not None:
            message += " chamfer:{}|{}|{}".format(green("{:.4f}".format(chamfer[0]), bold=True),
                                                    green("{:.4f}".format(chamfer[1]), bold=True),
                                                    green("{:.4f}".format((chamfer[0]+chamfer[1])/2), bold=True))
        print(message)
log = Log()

def update_timer(opt, timer, ep, it_per_ep):
    momentum = 0.99
    timer.elapsed = time.time()-timer.start
    timer.it = timer.it_end-timer.it_start
    # compute speed with moving average
    timer.it_mean = timer.it_mean*momentum+timer.it*(1-momentum) if timer.it_mean is not None else timer.it
    timer.arrival = timer.it_mean*it_per_ep*(opt.max_epoch-ep)

# move tensors to device in-place
def move_to_device(X, device):
    if isinstance(X, dict):
        for k, v in X.items():
            X[k] = move_to_device(v, device)
    elif isinstance(X, list):
        for i, e in enumerate(X):
            X[i] = move_to_device(e, device)
    elif isinstance(X, tuple) and hasattr(X, "_fields"): # collections.namedtuple
        dd = X._asdict()
        dd = move_to_device(dd, device)
        return type(X)(**dd)
    elif isinstance(X, torch.Tensor):
        return X.to(device=device)
    return X

# detach tensors
def detach_tensors(X):
    if isinstance(X, dict):
        for k, v in X.items():
            X[k] = detach_tensors(v)
    elif isinstance(X, list):
        for i, e in enumerate(X):
            X[i] = detach_tensors(e)
    elif isinstance(X, tuple) and hasattr(X, "_fields"): # collections.namedtuple
        dd = X._asdict()
        dd = detach_tensors(dd)
        return type(X)(**dd)
    elif isinstance(X, torch.Tensor):
        return X.detach()
    return X

# this recursion seems to only work for the outer loop when dict_type is not dict
def to_dict(D, dict_type=dict):
    D = dict_type(D)
    for k, v in D.items():
        if isinstance(v, dict):
            D[k] = to_dict(v, dict_type)
    return D

def get_child_state_dict(state_dict, key):
    out_dict = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            param_name = k[7:]
        else:
            param_name = k
        if param_name.startswith("{}.".format(key)):
            out_dict[".".join(param_name.split(".")[1:])] = v
    return out_dict

def restore_checkpoint(opt, model, load_name=None, resume=False, best=False, evaluate=False):
    assert((load_name is None)==(resume is not False)) # resume can be True/False or epoch numbers
    if resume:
        if best:
            load_name = "{0}/best.ckpt".format(opt.output_path)
        else:
            load_name = "{0}/latest.ckpt".format(opt.output_path) if resume==True else \
                        "{0}/checkpoint/ep{1}.ckpt".format(opt.output_path, opt.resume)
        checkpoint = torch.load(load_name, map_location=torch.device(opt.device))
        # strictly load all parameters
        if evaluate:
            pretrained_dict = checkpoint["graph"]
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "discriminator" not in k}
            missing, unexpected = model.graph.module.load_state_dict(pretrained_dict, strict=False)
            print('Missing keys:')
            print('########################')
            print(missing)
            print('########################')
            print('Unexpected keys:')
            print('########################')
            print(unexpected)
            print('########################')
        else:
            model.graph.module.load_state_dict(checkpoint["graph"], strict=True)            
    else:
        checkpoint = torch.load(load_name, map_location=torch.device(opt.device))
        # load individual (possibly partial) children modules
        for name, child in model.graph.module.named_children():
            child_state_dict = get_child_state_dict(checkpoint["graph"], name)
            if child_state_dict:
                print("restoring {} on device {}...".format(name, opt.device))
                child.load_state_dict(child_state_dict)
            else:
                print("skipping {} on device {}...".format(name, opt.device))
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched"] and key in checkpoint and resume:
            print("restoring {} on device {}...".format(key, opt.device))
            getattr(model, key).load_state_dict(checkpoint[key])
    if resume:
        if resume is not True: assert(resume==checkpoint["epoch"])
        ep, it, best_val = checkpoint["epoch"], checkpoint["iter"], checkpoint["best_val"]
        print("resuming from epoch {0} (iteration {1})".format(ep, it))
    else: ep, it, best_val = None, None, None
    return ep, it, best_val

def save_checkpoint(opt, model, ep, it, best_val, latest=False, best=False, children=None):
    os.makedirs("{0}/checkpoint".format(opt.output_path), exist_ok=True)
    if isinstance(model.graph, torch.nn.DataParallel) or isinstance(model.graph, torch.nn.parallel.DistributedDataParallel):
        graph = model.graph.module
    else:
        graph = model.graph
    if children is not None:
        graph_state_dict = { k: v for k, v in graph.state_dict().items() if k.startswith(children) }
    else: graph_state_dict = graph.state_dict()
    checkpoint = dict(
        epoch=ep,
        iter=it,
        best_val=best_val,
        graph=graph_state_dict,
    )
    for key in model.__dict__:
        if key.split("_")[0] in ["optim", "sched"]:
            checkpoint.update({ key: getattr(model, key).state_dict() })
    torch.save(checkpoint, "{0}/latest.ckpt".format(opt.output_path))
    if best:
        shutil.copy("{0}/latest.ckpt".format(opt.output_path),
                    "{0}/best.ckpt".format(opt.output_path))
    if not latest:
        shutil.copy("{0}/latest.ckpt".format(opt.output_path),
                    "{0}/checkpoint/ep{1}.ckpt".format(opt.output_path, ep))

def check_socket_open(hostname, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    is_open = False
    try:
        s.bind((hostname, port))
    except socket.error:
        is_open = True
    finally:
        s.close()
    return is_open

def get_layer_dims(layers):
    # return a list of tuples (k_in, k_out)
    return list(zip(layers[:-1], layers[1:]))

@contextlib.contextmanager
def suppress(stdout=False, stderr=False):
    with open(os.devnull, "w") as devnull:
        if stdout: old_stdout, sys.stdout = sys.stdout, devnull
        if stderr: old_stderr, sys.stderr = sys.stderr, devnull
        try: yield
        finally:
            if stdout: sys.stdout = old_stdout
            if stderr: sys.stderr = old_stderr

def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)

def compute_grad2(d_outs, x_in):
    d_outs = [d_outs] if not isinstance(d_outs, list) else d_outs
    reg = 0
    for d_out in d_outs:
        batch_size = x_in.size(0)
        grad_dout = torch.autograd.grad(
            outputs=d_out.sum(), inputs=x_in,
            create_graph=True, retain_graph=True, only_inputs=True
        )[0]
        grad_dout2 = grad_dout.pow(2)
        assert(grad_dout2.size() == x_in.size())
        reg += grad_dout2.view(batch_size, -1).sum(1)
    return reg / len(d_outs)

def compute_sampling_prob(opt, mask, uniform_fac=3):
    assert len(mask.shape) == 2
    h, w = mask.shape
    assert opt.H == h
    mask_binary = (mask > 0.5)
    # use boundaryDistanceTransform instead of distanceTransform (for 0.5 pixel precision)
    sdf_2D = vigra.filters.boundaryDistanceTransform(mask_binary.float().cpu().numpy())
    sdf_2D = torch.from_numpy(sdf_2D)
    prob_vec = 1 / (sdf_2D + uniform_fac)
    prob_vec = torch.nn.functional.normalize(prob_vec.view(h * w), dim=-1, p=1).cpu().numpy()
    indices = torch.tensor(np.random.choice(h * w, opt.render.rand_sample, p=prob_vec, replace=False))
    return indices

def setup(rank, world_size, port_no):
    full_address = 'tcp://127.0.0.1:' + str(port_no)
    dist.init_process_group("nccl", init_method=full_address, rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

def print_grad(grad, prefix=''):
    print("{} --- Grad Abs Mean, Grad Max, Grad Min: {:.5f} | {:.5f} | {:.5f}".format(prefix, grad.abs().mean().item(), grad.max().item(), grad.min().item()))

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class EasyDict(dict):
    def __init__(self, d=None, **kwargs):
        if d is None:
            d = {}
        else:
            d = dict(d)
        if kwargs:
            d.update(**kwargs)
        for k, v in d.items():
            setattr(self, k, v)
        # Class attributes
        for k in self.__class__.__dict__.keys():
            if not (k.startswith('__') and k.endswith('__')) and not k in ('update', 'pop'):
                setattr(self, k, getattr(self, k))

    def __setattr__(self, name, value):
        if isinstance(value, (list, tuple)):
            value = [self.__class__(x)
                     if isinstance(x, dict) else x for x in value]
        elif isinstance(value, dict) and not isinstance(value, self.__class__):
            value = self.__class__(value)
        super(EasyDict, self).__setattr__(name, value)
        super(EasyDict, self).__setitem__(name, value)

    __setitem__ = __setattr__

    def update(self, e=None, **f):
        d = e or dict()
        d.update(f)
        for k in d:
            setattr(self, k, d[k])

    def pop(self, k, d=None):
        delattr(self, k)
        return super(EasyDict, self).pop(k, d)

