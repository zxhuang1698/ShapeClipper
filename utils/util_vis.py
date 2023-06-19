import numpy as np
from sklearn.decomposition import PCA
import torch
import torch.nn.functional as torch_F
import torchvision
import torchvision.transforms.functional as torchvision_F
import matplotlib.pyplot as plt
import PIL
import PIL.ImageDraw
from PIL import Image, ImageFont
from utils.util import log
import seaborn as sn
import trimesh
import utils.camera as camera

@torch.no_grad()
def tb_image(opt, tb, step, group, name, images, masks=None, num_vis=None, from_range=(0, 1), poses=None, scales=None, cmap="gray"):
    images = preprocess_vis_image(opt, images, masks=masks, from_range=from_range, cmap=cmap)
    num_H, num_W = num_vis or opt.tb.num_images
    images = images[:num_H*num_W]
    if poses is not None:
        # poses: [B, 3, 4]
        # rots: [max(B, num_images), 3, 3]
        rots = poses[:num_H*num_W, ..., :3]
        images = torch.stack([draw_pose(opt, image, rot, size=20, width=2) for image, rot in zip(images, rots)], dim=0)
    if scales is not None:
        images = torch.stack([draw_scale(opt, image, scale.item()) for image, scale in zip(images, scales)], dim=0)
    image_grid = torchvision.utils.make_grid(images[:, :3], nrow=num_W, pad_value=1.)
    if images.shape[1]==4:
        mask_grid = torchvision.utils.make_grid(images[:, 3:], nrow=num_W, pad_value=1.)[:1]
        image_grid = torch.cat([image_grid, mask_grid], dim=0)
    tag = "{0}/{1}".format(group, name)
    tb.add_image(tag, image_grid, step)

def preprocess_vis_image(opt, images, masks=None, from_range=(0, 1), cmap="gray"):
    min, max = from_range
    images = (images-min)/(max-min)
    if masks is not None:
        # then the mask is directly the transparency channel of png
        images = torch.cat([images, masks], dim=1)
    images = images.clamp(min=0, max=1).cpu()
    if images.shape[1]==1:
        images = get_heatmap(opt, images[:, 0].cpu(), cmap=cmap)
    return images

def preprocess_depth_image(opt, depth, mask=None, max_depth=1000):
    if mask is not None: depth = depth * mask + (1 - mask) * max_depth  # min of this will leads to minimum of masked regions
    depth = depth - depth.min()
    
    if mask is not None: depth = depth * mask   # max of this will leads to maximum of masked regions
    depth = depth / depth.max()
    return depth

def dump_images(opt, idx, name, images, masks=None, from_range=(0, 1), poses=None, scales=None, cmap="gray", folder='dump'):
    images = preprocess_vis_image(opt, images, masks=masks, from_range=from_range, cmap=cmap) # [B, 3, H, W]
    if poses is not None:
        rots = poses[..., :3]
        images = torch.stack([draw_pose(opt, image, rot, size=20, width=2) for image, rot in zip(images, rots)], dim=0)
    if scales is not None:
        images = torch.stack([draw_scale(opt, image, scale.item()) for image, scale in zip(images, scales)], dim=0)
    images = images.cpu().permute(0, 2, 3, 1).contiguous().numpy() # [B, H, W, 3]
    for i, img in zip(idx, images):
        fname = "{}/{}/{}_{}.png".format(opt.output_path, folder, i, name)
        img = Image.fromarray((img*255).astype(np.uint8))
        img.save(fname)

# img_list is a list of length n_views, where each view is a image tensor of [B, 3, H, W] 
def dump_gifs(opt, idx, name, imgs_list, from_range=(0, 1), folder='dump', cmap="gray"):
    for i in range(len(imgs_list)):
        imgs_list[i] = preprocess_vis_image(opt, imgs_list[i], from_range=from_range, cmap=cmap)
    for i in range(len(idx)):
        img_list_np = [imgs[i].cpu().permute(1, 2, 0).contiguous().numpy() for imgs in imgs_list]  # list of [H, W, 3], each item is a view of ith sample
        img_list_pil = [Image.fromarray((img*255).astype(np.uint8)).convert('RGB') for img in img_list_np]
        fname = "{}/{}/{}_{}.gif".format(opt.output_path, folder, idx[i], name)
        img_list_pil[0].save(fname, format='GIF', append_images=img_list_pil[1:], save_all=True, duration=100, loop=0)

def get_heatmap(opt, gray, cmap): # [N, H, W]
    color = plt.get_cmap(cmap)(gray.numpy())
    color = torch.from_numpy(color[..., :3]).permute(0, 3, 1, 2).contiguous().float() # [N, 3, H, W]
    return color

def dump_meshes(opt, idx, name, meshes, folder='dump'):
    for i, mesh in zip(idx, meshes):
        fname = "{}/{}/{}_{}.ply".format(opt.output_path, folder, i, name)
        try:
            mesh.export(fname)
        except:
            print('Mesh is empty!')

def dump_pointclouds_compare(opt, idx, name, preds, gts, folder='dump'):
    for i in range(len(idx)):
        pred = preds[i].cpu().numpy()   # [N, 3]
        gt = gts[i].cpu().numpy()   # [N, 3]
        color_pred = np.zeros(pred.shape).astype(np.uint8)
        color_pred[:, 0] = 255
        color_gt = np.zeros(gt.shape).astype(np.uint8)
        color_gt[:, 1] = 255
        pc_vertices = np.vstack([pred, gt])
        colors = np.vstack([color_pred, color_gt])
        pc_color = trimesh.points.PointCloud(vertices=pc_vertices, colors=colors)
        fname = "{}/{}/{}_{}.ply".format(opt.output_path, folder, idx[i], name)
        pc_color.export(fname)

def dump_pointclouds(opt, idx, name, pcs, colors, folder='dump'):
    for i, pc, color in zip(idx, pcs, colors):
        pc = pc.cpu().numpy()   # [B, N, 3]
        pc_color = trimesh.points.PointCloud(vertices=pc, colors=color)
        fname = "{}/{}/{}_{}.ply".format(opt.output_path, folder, i, name)
        pc_color.export(fname)

@torch.no_grad()
def draw_pose(opt, image, rot_mtrx, size=15, width=1):
    # rot_mtrx: [3, 4]
    mode = "RGBA" if image.shape[0]==4 else "RGB"
    image_pil = torchvision_F.to_pil_image(image.cpu()).convert("RGBA")
    draw_pil = PIL.Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(draw_pil)
    center = (size, size)
    # first column of rotation matrix is the rotated vector of [1, 0, 0]'
    # second column of rotation matrix is the rotated vector of [0, 1, 0]'
    # third column of rotation matrix is the rotated vector of [0, 0, 1]'
    # then always take the first two element of each column is a projection to the 2D plane for visualization
    endpoint = [(size+size*p[0], size+size*p[1]) for p in rot_mtrx.t()]
    draw.line([center, endpoint[0]], fill=(255, 0, 0), width=width)
    draw.line([center, endpoint[1]], fill=(0, 255, 0), width=width)
    draw.line([center, endpoint[2]], fill=(0, 0, 255), width=width)
    image_pil.alpha_composite(draw_pil)
    image_drawn = torchvision_F.to_tensor(image_pil.convert(mode))
    return image_drawn

@torch.no_grad()
def draw_scale(opt, image, scale):
    mode = "RGBA" if image.shape[0]==4 else "RGB"
    image_pil = torchvision_F.to_pil_image(image.cpu()).convert("RGBA")
    draw_pil = PIL.Image.new("RGBA", image_pil.size, (0, 0, 0, 0))
    draw = PIL.ImageDraw.Draw(draw_pil)
    font = ImageFont.truetype("DejaVuSans.ttf", 9)
    position = (image_pil.size[0] - 30, image_pil.size[1] - 12)
    draw.text(position, '{:.3f}'.format(scale), fill="green", font=font) 
    image_pil.alpha_composite(draw_pil)
    image_drawn = torchvision_F.to_tensor(image_pil.convert(mode))
    return image_drawn