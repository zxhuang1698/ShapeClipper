import numpy as np
import torch
import torch.nn.functional as torch_F
import torchvision.transforms.functional as torchvision_F
import PIL
import json

from . import base
from utils.util import EasyDict as edict, compute_sampling_prob
import utils.camera as camera
import csv

class Dataset(base.Dataset):

    def __init__(self, opt, split="train", transform=None):
        super().__init__(opt, split)
        
        self.cat_id_all = dict(
            bed='bed', 
            bookcase='bookcase', 
            chair='chair', 
            desk='desk', 
            misc='misc', 
            sofa='sofa', 
            table='table', 
            tool='tool', 
            wardrobe='wardrobe'
        )
        # when transform is given, the dataset will be used for clip annotation
        self.clip_anno = transform is not None
        self.transform = transform
        self.max_imgs = opt.data.max_img_cat if opt.data.max_img_cat is not None else np.inf
        self.cat2label = {}
        accum_idx = 0
        self.cat_id = list(self.cat_id_all.values()) if opt.data.pix3d.cat is None else \
                      [v for k, v in self.cat_id_all.items() if k in opt.data.pix3d.cat.split(",")]
        for cat in self.cat_id:
            self.cat2label[cat] = accum_idx
            accum_idx += 1
        self.label2cat = []
        for cat in self.cat_id:
            key = next(key for key, value in self.cat_id_all.items() if value == cat)
            self.label2cat.append(key) 
            
        self.path = "data/Pix3D"
        self.list = self.get_list(opt, split)
        if self.clip_anno:
            self.get_path_list(opt)
        else:
            self.NN_dict = self.get_NN_anno(opt)
    
    # read the list file, return a list of tuple, (category, sample_name)
    def get_list(self, opt, split):
        cads = []
        for c in self.cat_id:
            list_fname = "{}/lists/{}_{}.txt".format(self.path, c, split)
            for i, m in enumerate(open(list_fname).read().splitlines()):
                if i >= self.max_imgs: break
                cads.append((c, m))
        return cads

    def get_path_list(self, opt):
        self.img_path_list = []
        self.pc_path_list = []
        self.rel_path_list = []
        for idx in range(len(self.list)):
            meta = self.get_metadata(opt, idx)
            pc_fname = "{0}/{1}".format(self.path, "pointclouds/" + meta.cad_path[6:])
            pc_fname = pc_fname.replace(".obj", ".npy")
            image_fname = "{0}/{1}".format(self.path, meta.img_path)
            self.pc_path_list.append(pc_fname)
            self.img_path_list.append(image_fname)
            self.rel_path_list.append('/'.join(meta.img_path.split('/')[1:]))

    def name_from_path(self, opt, relpath):
        c = relpath.split('/')[0]
        name = relpath.split('/')[1].split('.')[0]
        return c, name

    def id_filename_mapping(self, opt, outpath):
        outfile = open(outpath, 'w')
        for i in range(len(self.list)):
            meta = self.get_metadata(opt, i)
            image_fname = "{0}/{1}".format(self.path, meta.img_path)
            mask_fname = "{0}/{1}".format(self.path, meta.mask_path)
            normal_path = meta.mask_path.replace("mask", "normal")
            normal_fname = "{0}/{1}".format(self.path, normal_path)
            pc_fname = "{0}/{1}".format(self.path, "pointclouds/" + meta.cad_path[6:])
            pc_fname = pc_fname.replace(".obj", ".npy")
            outfile.write("{} {} {} {} {}\n".format(i, image_fname, mask_fname, normal_fname, pc_fname))
        outfile.close()
        
    # read the NN annotation as a list, convert it as a dictionary
    # keys are tuple of category/img_name, values are list of tuples
    def get_NN_anno(self, opt):
        dict_anno = {}
        category_name = opt.data[opt.data.dataset].cat.replace(', ', '_')
        NN_fname = "{}/CLIP_NN/{}_{}.csv".format(self.path, category_name, self.split)
        with open(NN_fname, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            list_anno = list(csvreader)[1:]
        for anno in list_anno:
            c, name = self.name_from_path(opt, anno[0])
            dict_anno[(c, name)] = []
            for nearest in anno[1:1+opt.data.k_nearest]:
                nearest_c, nearest_name = self.name_from_path(opt, nearest)
                dict_anno[(c, name)].append((nearest_c, nearest_name))
        return dict_anno
    
    def __getitem__(self, idx):
        opt = self.opt
        sample = dict(idx=idx)
        
        # load meta
        meta = self.get_metadata(opt, idx)
        
        # early return if clip annotation
        if self.clip_anno:
            image = self.get_image(opt, meta=meta)
            rgb_input_map, _ = self.preprocess_image(opt, image, transform=self.transform)
            sample.update(rgb_input=rgb_input_map)
            return sample

        # load images and compute distance transform
        image = self.get_image(opt, meta=meta)
        cat_label, _ = self.get_category(opt, idx)
        rgb_input_map, mask_input_map = self.preprocess_image(opt, image)
        normal_input_map = self.get_normal(opt, meta, mask_input_map)
        sample.update(
            rgb_input_map=rgb_input_map,
            mask_input_map=mask_input_map,
            normal_input_map=normal_input_map,
            category_label=cat_label,
        )

        rgb_input, mask_input, normal_input, ray_idx = \
            self.sample_map(opt, rgb_input_map, mask_input_map, normal_input_map)
        sample.update(
            rgb_input=rgb_input,
            mask_input=mask_input,
            normal_input=normal_input,
        )
        if ray_idx is not None:
            sample.update(ray_idx=ray_idx)
        
        # load pose
        intr, pose = self.get_camera(opt, meta=meta)
        sample.update(
            pose_gt=pose,
            intr=intr,
        )
        
        # load GT point cloud (only for evaluation)
        dpc = self.get_pointcloud(opt, idx, meta=meta)
        sample.update(dpc=dpc)

        # load nearest neighbors
        c, name = self.list[idx]
        neighbors = self.NN_dict[(c, name)]
        
        # list to collect maps
        rgb_input_map_NN_list = []
        mask_input_map_NN_list = []
        normal_input_map_NN_list = []
        
        # list to collect tensors
        rgb_input_NN_list = []
        mask_input_NN_list = []
        normal_input_NN_list = []
        ray_idx_NN_list = []
        pose_NN_list = []
        
        # loop over neighbors
        for i in range(opt.data.k_nearest):
            c_n, name_n = neighbors[i]
            meta_n = self.get_metadata(opt, 0, name_n, c_n)
            input_NN = self.get_NN(opt, meta_n, c_n)

            # get the input maps
            rgb_input_map_NN = input_NN.rgb_input_map
            rgb_input_map_NN_list.append(rgb_input_map_NN)
            mask_input_map_NN = input_NN.mask_input_map
            mask_input_map_NN_list.append(mask_input_map_NN)
            normal_input_map_NN = input_NN.normal_input_map
            normal_input_map_NN_list.append(normal_input_map_NN)

            # get the sampled tensor
            rgb_input_NN, mask_input_NN, normal_input_NN, ray_idx_NN = \
                self.sample_map(opt, rgb_input_map_NN, mask_input_map_NN, normal_input_map_NN)
                
            # get pose
            pose_NN = self.get_camera(opt, meta=meta)[1]
            
            # collect into lists
            if ray_idx_NN is not None:
                ray_idx_NN_list.append(ray_idx_NN)
            rgb_input_NN_list.append(rgb_input_NN)
            mask_input_NN_list.append(mask_input_NN)
            normal_input_NN_list.append(normal_input_NN)
            pose_NN_list.append(pose_NN)

        # stack the lists to tensors
        rgb_input_NN = torch.stack(rgb_input_NN_list, dim=-1)
        mask_input_NN = torch.stack(mask_input_NN_list, dim=-1)
        normal_input_NN = torch.stack(normal_input_NN_list, dim=-1)
        pose_NN = torch.stack(pose_NN_list, dim=-1)
        
        rgb_input_map_NN = torch.stack(rgb_input_map_NN_list, dim=-1)
        mask_input_map_NN = torch.stack(mask_input_map_NN_list, dim=-1)
        normal_input_map_NN = torch.stack(normal_input_map_NN_list, dim=-1)
        if len(ray_idx_NN_list) > 0:
            ray_idx_NN = torch.stack(ray_idx_NN_list, dim=-1)
            
        # prepare the input dict
        sample.update(
            rgb_input_NN=rgb_input_NN,
            mask_input_NN=mask_input_NN,
            normal_input_NN=normal_input_NN,
            rgb_input_map_NN=rgb_input_map_NN,
            mask_input_map_NN=mask_input_map_NN,
            normal_input_map_NN=normal_input_map_NN,
            pose_gt_NN=pose_NN
        )
        if len(ray_idx_NN_list) > 0:
            sample.update(
                ray_idx_NN=ray_idx_NN,
            )
        return sample

    def sample_map(self, opt, rgb_map, mask_map, normal_map):
        rgb = rgb_map.permute(1,2,0).view(opt.H*opt.W,3)
        mask = mask_map.permute(1,2,0).view(opt.H*opt.W,1)
        normal = normal_map.permute(1,2,0).view(opt.H*opt.W,3)
        ray_idx = None
        if self.split=="train" and opt.render.rand_sample:
            ray_idx = compute_sampling_prob(opt, mask_map[0], opt.render.ray_uniform_fac)
            rgb, mask = rgb[ray_idx], mask[ray_idx]
            normal = normal[ray_idx]
        return rgb, mask, normal, ray_idx

    def get_NN(self, opt, meta, category):
        input_NN = edict()
        image = self.get_image(opt, meta=meta)
        rgb, mask = self.preprocess_image(opt, image)
        normal = self.get_normal(opt, meta, mask)
        input_NN.update(
            rgb_input_map=rgb,
            mask_input_map=mask,
            normal_input_map=normal,
        )
        return input_NN

    def get_image(self, opt, meta):
        image_fname = "{0}/{1}".format(self.path, meta.img_path)
        image = PIL.Image.open(image_fname).convert("RGB")
        mask_fname = "{0}/{1}".format(self.path, meta.mask_path)
        mask = PIL.Image.open(mask_fname).convert("L")
        image = PIL.Image.merge("RGBA", (*image.split(), mask))
        return image
    
    def get_normal(self, opt, meta, mask):
        normal_path = meta.mask_path.replace("mask", "normal")
        normal_fname = "{0}/{1}".format(self.path, normal_path)
        normal = PIL.Image.open(normal_fname).convert("RGB")
        normal = normal.resize((opt.W, opt.H))
        normal = torchvision_F.to_tensor(normal)
        assert normal.shape[0] == 3
        normal = (normal - 0.5) * 2
        normal = torch_F.normalize(normal, dim=0, p=2)
        normal = normal * mask
        return normal

    def get_category(self, opt, idx):
        c, _ = self.list[idx]
        label = int(self.cat2label[c])
        return label, c

    def preprocess_image(self, opt, image, transform=None):
        image = image.resize((opt.W, opt.H))
        image = torchvision_F.to_tensor(image)
        rgb, mask = image[:3], image[3:]
        mask = (mask>0.5).float()
        if opt.data.bgcolor is not None:
            # replace background color using mask
            rgb = rgb*mask+opt.data.bgcolor*(1-mask)
        if transform is not None:
            image_pil = torchvision_F.to_pil_image(rgb)
            rgb = transform(image_pil)
        return rgb, mask

    def get_camera(self, opt, meta=None):
        intr = torch.tensor([[opt.camera.focal*opt.W, 0, opt.W/2],
                             [0, opt.camera.focal*opt.H, opt.H/2],
                             [0, 0, 1]])
        R_raw = meta.cam.R
        R_trans = torch.tensor([
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ]).float().to(R_raw.device)
        R = torch.mm(R_trans, R_raw)
        pose_R = camera.pose(R=R)
        pose_T= camera.pose(t=[0, 0, opt.camera.dist])
        pose = camera.pose.compose([pose_R, pose_T])
        return intr, pose

    def get_pointcloud(self, opt, idx, meta=None):
        pc_fname = "{0}/{1}".format(self.path, "pointclouds/" + meta.cad_path[6:])
        pc_fname = pc_fname.replace(".obj", ".npy")
        pc = torch.from_numpy(np.load(pc_fname)).float()
        dpc = dict(
            points=pc,
            normals=torch.zeros_like(pc),
        )
        return dpc

    def square_crop(self, opt, image, cat, bbox=None, crop_ratio=1.):
        # crop to canonical image size
        x1, y1, x2, y2 = bbox
        h, w = y2-y1, x2-x1
        yc, xc = (y1+y2)/2, (x1+x2)/2
        S = max(h, w)*1.2
        # crop with random size (cropping out of boundary = padding)
        S2 = S*crop_ratio
        image = torchvision_F.crop(image, top=int(yc-S2/2), left=int(xc-S2/2), height=int(S2), width=int(S2))
        return image

    def get_metadata(self, opt, idx, name=None, c=None):
        if name is None or c is None:
            c, name = self.list[idx]
        meta_fname = "{}/annotation/{}/{}.json".format(self.path, c, name)
        meta = json.load(open(meta_fname, "r", encoding='utf-8'))
        img_path = meta["img"].replace("img", "img_processed")
        mask_path = meta["mask"].replace("mask", "mask_processed")
        meta_out = edict(
            cam=edict(
                focal=float(meta["focal_length"]),
                cam_loc=torch.tensor(meta["cam_position"]),
                R=torch.tensor(meta["rot_mat"]),
                T=torch.tensor(meta["trans_mat"]),
            ),
            img_path=img_path,
            mask_path=mask_path,
            cad_path=meta["model"],
            bbox=torch.tensor(meta["bbox"]),
        )
        return meta_out

    def __len__(self):
        return len(self.list)
