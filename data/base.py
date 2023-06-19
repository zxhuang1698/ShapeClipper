import numpy as np
import torch
import torchvision
import PIL
from utils.util import EasyDict as edict
from copy import deepcopy

class Dataset(torch.utils.data.Dataset):

    def __init__(self, opt, split):
        super().__init__()
        self.opt = deepcopy(opt)
        self.split = split
        self.augment = split=="train" and opt.data.augment

    def setup_loader(self, opt, shuffle=False, drop_last=True, subcat=None, batch_size=None, allow_ddp=True):
        sampler = torch.utils.data.distributed.DistributedSampler(self,
            num_replicas=opt.world_size, rank=opt.device
        ) if (self.split == "train" and allow_ddp and 'world_size' in opt) else None
        if batch_size is None: batch_size=opt.batch_size
        loader = torch.utils.data.DataLoader(self,
            batch_size=batch_size,
            num_workers=opt.data.num_workers,
            shuffle=shuffle if sampler is None else False,
            drop_last=drop_last,
            sampler=sampler
        )
        if opt.device == 0:
            print("number of samples: {}".format(len(self)))
        return loader

    def get_list(self, opt):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError

    def get_image(self, opt, idx):
        raise NotImplementedError

    def generate_augmentation(self, opt):
        brightness = opt.data.augment.brightness or 0.
        contrast = opt.data.augment.contrast or 0.
        saturation = opt.data.augment.saturation or 0.
        hue = opt.data.augment.hue or 0.
        color_jitter = torchvision.transforms.ColorJitter(
            brightness=(1-brightness, 1+brightness),
            contrast=(1-contrast, 1+contrast),
            saturation=(1-saturation, 1+saturation),
            hue=(-hue, hue),
        )
        aug = edict(
            color_jitter=color_jitter,
            flip=np.random.randn()>0 if opt.data.augment.hflip else False,
            crop_ratio=1+(np.random.rand()*2-1)*opt.data.augment.crop_scale if opt.data.augment.crop_scale else 1,
            rot_angle=(np.random.rand()*2-1)*opt.data.augment.rotate if opt.data.augment.rotate else 0,
        )
        return aug

    def apply_color_jitter(self, opt, image, color_jitter):
        mode = image.mode
        if mode!="L":
            chan = image.split()
            rgb = PIL.Image.merge("RGB", chan[:3])
            rgb = color_jitter(rgb)
            rgb_chan = rgb.split()
            image = PIL.Image.merge(mode, rgb_chan+chan[3:])
        return image

    def __len__(self):
        return len(self.list)
