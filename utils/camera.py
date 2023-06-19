import numpy as np
import torch
import torch.nn.functional as torch_F

class Pose():
    # a pose class with util methods
    def __call__(self, R=None, t=None):
        assert(R is not None or t is not None)
        if R is None:
            if not isinstance(t, torch.Tensor): t = torch.tensor(t)
            R = torch.eye(3, device=t.device).repeat(*t.shape[:-1], 1, 1)
        elif t is None:
            if not isinstance(R, torch.Tensor): R = torch.tensor(R)
            t = torch.zeros(R.shape[:-1], device=R.device)
        else:
            if not isinstance(R, torch.Tensor): R = torch.tensor(R)
            if not isinstance(t, torch.Tensor): t = torch.tensor(t)
        assert(R.shape[:-1]==t.shape and R.shape[-2:]==(3, 3))
        R = R.float()
        t = t.float()
        pose = torch.cat([R, t[..., None]], dim=-1) # [..., 3, 4]
        assert(pose.shape[-2:]==(3, 4))
        return pose

    def invert(self, pose, use_inverse=False):
        R, t = pose[..., :3], pose[..., 3:]
        R_inv = R.inverse() if use_inverse else R.transpose(-1, -2)
        t_inv = (-R_inv@t)[..., 0]
        pose_inv = self(R=R_inv, t=t_inv)
        return pose_inv

    def compose(self, pose_list):
        # pose_new(x) = poseN(...(pose2(pose1(x)))...)
        pose_new = pose_list[0]
        for pose in pose_list[1:]:
            pose_new = self.compose_pair(pose_new, pose)
        return pose_new

    def compose_pair(self, pose_a, pose_b):
        # pose_new(x) = pose_b(pose_a(x))
        R_a, t_a = pose_a[..., :3], pose_a[..., 3:]
        R_b, t_b = pose_b[..., :3], pose_b[..., 3:]
        R_new = R_b@R_a
        t_new = (R_b@t_a+t_b)[..., 0]
        pose_new = self(R=R_new, t=t_new)
        return pose_new

pose = Pose()

def get_transformed_grid(opt, points_3D, pose, pose_gt):
    points_3D_cam = world2cam(points_3D, pose_gt.unsqueeze(1).unsqueeze(1))
    points_3D_transformed = cam2world(points_3D_cam, pose.unsqueeze(1).unsqueeze(1))
    return points_3D_transformed

def pose_from_azim_elev(azim, elev):
    cos_a, sin_a = azim[:, 0], azim[:, 1]
    cos_e, sin_e = elev[:, 0], elev[:, 1]
    x = cos_a * cos_e
    y = sin_a * cos_e
    z = sin_e
    cam_location = torch.stack([x, y, z], dim=-1)
    forward = -cam_location
    tmp = torch.tensor([[0., 0., -1]]).to(azim.device).expand_as(forward)
    
    right = torch.cross(tmp, forward)
    right = torch_F.normalize(right, dim=-1, p=2)
    
    up = torch.cross(forward, right)
    up = torch_F.normalize(up, dim=-1, p=2)     # [B, 3]
    
    rotation = torch.stack([right, up, forward], dim=-1).permute(0, 2, 1).contiguous()
    
    # [B, 3, 3]
    return rotation

def to_hom(X):
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)
    return X_hom

def world2cam(X, pose): # [B, N, 3]
    X_hom = to_hom(X)
    return X_hom@pose.transpose(-1, -2)

def cam2img(X, cam_intr):
    return X@cam_intr.transpose(-1, -2)

def img2cam(X, cam_intr):
    return X@cam_intr.inverse().transpose(-1, -2)

def cam2world(X, pose):
    X_hom = to_hom(X)
    pose_inv = Pose().invert(pose)
    # pose_inv.transpose(-1, -2): [B, 4, 3]
    # originally, p_cam' = [R|t] @ [p_world, 1]'
    # therefore, p_world = [p_cam, 1] @ inv([R|t])'
    return X_hom@pose_inv.transpose(-1, -2)

def transform_normal(normals, pose):
    rotation = pose[:, :, :3]
    translation = torch.zeros(1, 3, 1, device=rotation.device).expand(rotation.shape[0], 3, 1)
    normal_transform = torch.cat([rotation, translation], dim=-1)
    normals_transformed = cam2world(normals, normal_transform)
    return normals_transformed

def azim_to_rotation_matrix(azim, representation='rad'):
    """Azim is angle with vector +X, rotated in XZ plane"""
    if representation == 'rad':
        # [B, ]
        cos, sin = torch.cos(azim), torch.sin(azim)
    elif representation == 'angle':
        # [B, ]
        azim = azim * np.pi / 180
        cos, sin = torch.cos(azim), torch.sin(azim)
    elif representation == 'trig':
        # [B, 2]
        cos, sin = azim[:, 0], azim[:, 1]
    R = torch.eye(3, device=azim.device)[None].repeat(len(azim), 1, 1)
    zeros = torch.zeros(len(azim), device=azim.device)
    R[:, 0, :] = torch.stack([cos, zeros, sin], dim=-1)
    R[:, 2, :] = torch.stack([-sin, zeros, cos], dim=-1)
    return R

def elev_to_rotation_matrix(elev, representation='rad'):
    """Angle with vector +Z in YZ plane"""
    if representation == 'rad':
        # [B, ]
        cos, sin = torch.cos(elev), torch.sin(elev)
    elif representation == 'angle':
        # [B, ]
        elev = elev * np.pi / 180
        cos, sin = torch.cos(elev), torch.sin(elev)
    elif representation == 'trig':
        # [B, 2]
        cos, sin = elev[:, 0], elev[:, 1]
    R = torch.eye(3, device=elev.device)[None].repeat(len(elev), 1, 1)
    R[:, 1, 1:] = torch.stack([cos, -sin], dim=-1)
    R[:, 2, 1:] = torch.stack([sin, cos], dim=-1)
    return R

def roll_to_rotation_matrix(roll, representation='rad'):
    """Angle with vector +X in XY plane"""
    if representation == 'rad':
        # [B, ]
        cos, sin = torch.cos(roll), torch.sin(roll)
    elif representation == 'angle':
        # [B, ]
        roll = roll * np.pi / 180
        cos, sin = torch.cos(roll), torch.sin(roll)
    elif representation == 'trig':
        # [B, 2]
        cos, sin = roll[:, 0], roll[:, 1]
    R = torch.eye(3, device=roll.device)[None].repeat(len(roll), 1, 1)
    R[:, 0, :2] = torch.stack([cos, sin], dim=-1)
    R[:, 1, :2] = torch.stack([-sin, cos], dim=-1)
    return R

def get_camera_grid(opt,batch_size,device,intr=None):
    # compute image coordinate grid
    if opt.camera.model=="perspective":
        y_range = torch.arange(opt.H,dtype=torch.float32,device=device).add_(0.5)
        x_range = torch.arange(opt.W,dtype=torch.float32,device=device).add_(0.5)
        Y,X = torch.meshgrid(y_range,x_range) # [H,W]
        xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
    elif opt.camera.model=="orthographic":
        assert(opt.H==opt.W)
        y_range = torch.linspace(-1,1,opt.H,device=device)
        x_range = torch.linspace(-1,1,opt.W,device=device)
        Y,X = torch.meshgrid(y_range,x_range) # [H,W]
        xy_grid = torch.stack([X,Y],dim=-1).view(-1,2) # [HW,2]
    xy_grid = xy_grid.repeat(batch_size,1,1) # [B,HW,2]
    if opt.camera.model=="perspective":
        grid_3D = img2cam(to_hom(xy_grid),intr) # [B,HW,3]
    elif opt.camera.model=="orthographic":
        grid_3D = to_hom(xy_grid) # [B,HW,3]
    return xy_grid,grid_3D

def get_center_and_ray(opt,pose,intr=None,offset=None,device=None): # [HW,2]
    if device is None: device = opt.device
    batch_size = len(pose)
    # grid 3D is the 3D location of the 2D pixels (on image plane, d=1)
    # under camera frame
    xy_grid,grid_3D = get_camera_grid(opt,batch_size,device,intr=intr) # [B,HW,3]
    # compute center and ray
    if opt.camera.model=="perspective":
        if offset is not None:
            grid_3D[...,:2] += offset
        # camera pose, [0, 0, 0] under camera frame
        center_3D = torch.zeros(batch_size,1,3,device=xy_grid.device) # [B,1,3]
    elif opt.camera.model=="orthographic":
        # different ray has different camera center
        center_3D = torch.cat([xy_grid,torch.zeros_like(xy_grid[...,:1])],dim=-1) # [B,HW,3]
    # transform from camera to world coordinates
    grid_3D = cam2world(grid_3D,pose) # [B,HW,3]
    center_3D = cam2world(center_3D,pose) # [B,HW,3]
    ray = grid_3D-center_3D # [B,HW,3]
    return center_3D,ray

def get_intr(opt, scale_focal):
    zeros = torch.zeros_like(scale_focal)
    ones = torch.ones_like(scale_focal)
    f = opt.camera.focal * scale_focal
    batch_size = scale_focal.shape[0]
    intr = torch.stack(
        [
            f * opt.W, zeros, ones * opt.W / 2,
            zeros, f * opt.H, ones * opt.H / 2,
            zeros, zeros, ones
        ], 
        dim=-1
    ).view(batch_size, 3, 3).contiguous()
    return intr