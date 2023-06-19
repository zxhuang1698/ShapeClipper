import numpy as np
import torch
import threading
import mcubes
import trimesh
import chamfer_3D

@torch.no_grad()
def get_dense_3D_grid(opt, var, N=None):
    batch_size = len(var.idx)
    N = N or opt.eval.vox_res
    # -0.6, 0.6
    range_min, range_max = opt.eval.range
    grid = torch.linspace(range_min, range_max, N+1, device=opt.device)
    points_3D = torch.stack(torch.meshgrid(grid, grid, grid), dim=-1) # [N, N, N, 3]
    # actually N+1 instead of N
    points_3D = points_3D.repeat(batch_size, 1, 1, 1, 1) # [B, N, N, N, 3]
    return points_3D

@torch.no_grad()
def compute_level_grid(opt, sdf_network, proj_latent_sdf, points_3D):
    # process points in sliced way
    batch_size = points_3D.shape[0]
    N = points_3D.shape[1]

    level_all = []
    slice_batch_size = 1
    for i in range(0,N,slice_batch_size):
        # [B, 1, N, N, 3]
        points_3D_batch = points_3D[:,i:i+slice_batch_size]
        # [B * N * N, 3]
        points_flat = points_3D_batch.reshape(-1, 3)
        # [B * N * N, 1]
        level_batch = sdf_network.get_conditional_output(opt, batch_size, points_flat, proj_latent_sdf, compute_grad=False)[0]
        level_all.append(level_batch.view(batch_size, 1, N, N, 1))

    level = torch.cat(level_all,dim=1)[...,0]
    return level

@torch.no_grad()
def normalize_pc(pc):
    assert len(pc.shape) == 3
    pc_mean = pc.mean(dim=1, keepdim=True) 
    pc_zmean = pc - pc_mean
    length_x = pc_zmean[:, :, 0].max(dim=-1)[0] - pc_zmean[:, :, 0].min(dim=-1)[0]
    length_y = pc_zmean[:, :, 1].max(dim=-1)[0] - pc_zmean[:, :, 1].min(dim=-1)[0]
    length_max = torch.stack([length_x, length_y], dim=-1).max(dim=-1)[0].unsqueeze(-1).unsqueeze(-1)
    pc_normalized = pc_zmean / (length_max + 1.e-7)
    return pc_normalized

@torch.no_grad()
def eval_metrics(opt, var, sdf_network, vis_only=False):
    points_3D = get_dense_3D_grid(opt, var) # [B, N, N, N, 3]
    batch_size = points_3D.shape[0]
    level_vox = compute_level_grid(opt, sdf_network, var.proj_latent_sdf, points_3D) # [B, N, N, N], [B, N, N, N, 3]
    var.eval_vox = points_3D.view(batch_size, -1, 3)
    # occ_grids: a list of length B, each is [N, N, N]
    *level_grids, = level_vox.cpu().numpy()
    meshes,pointclouds = convert_to_explicit(opt,level_grids,isoval=0.,to_pointcloud=True)
    var.mesh_pred = meshes
    var.dpc_pred = torch.tensor(pointclouds, dtype=torch.float32, device=opt.device)
    if opt.data.dataset in ['openimage']:
        var.f_score = torch.zeros(batch_size, len(opt.eval.f_thresholds)).to(var.idx.device)
        var.cd_acc = torch.zeros(batch_size).to(var.idx.device)
        var.cd_comp = torch.zeros(batch_size).to(var.idx.device)
        if vis_only: 
            return
        return torch.tensor(0).to(var.idx.device), torch.tensor(0).to(var.idx.device)
    
    # transform the prediction to view-centered frame
    R_pred = var.pose[..., :3]
    var.dpc_pred = (R_pred @ var.dpc_pred.permute(0, 2, 1)).permute(0, 2, 1).contiguous() 
    # transform the gt to view-centered frame
    R_gt = var.pose_gt[..., :3]
    var.dpc.points = (R_gt @ var.dpc.points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
    # rotate all the points into view-centric frames
    # we also do scale adjustment based on the crop scale
    if opt.data.dataset in ['pix3d']:
        R_trans_pred = torch.tensor([
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, -1]
        ]).float().to(R_pred.device).unsqueeze(0).expand_as(R_pred)
        var.dpc_pred = (R_trans_pred @ var.dpc_pred.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
        R_trans_gt = torch.tensor([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, 1]
        ]).float().to(R_gt.device).unsqueeze(0).expand_as(R_gt)
        var.dpc.points = (R_trans_gt @ var.dpc.points.permute(0, 2, 1)).permute(0, 2, 1).contiguous()
    var.dpc_pred = normalize_pc(var.dpc_pred)
    var.dpc.points = normalize_pc(var.dpc.points)

    if vis_only: 
        return
    dist_acc, dist_comp, _, _ = chamfer_distance(opt, X1=var.dpc_pred, X2=var.dpc.points)
    var.f_score = compute_fscore(dist_acc, dist_comp, opt.eval.f_thresholds)   # [B, n_threshold]
    # dist_acc: [B, n_points_pred]
    # dist_comp: [B, n_points_gt]
    assert dist_acc.shape[1] == opt.eval.num_points
    var.cd_acc = dist_acc.mean(dim=1)
    var.cd_comp = dist_comp.mean(dim=1)
    return dist_acc.mean(), dist_comp.mean()

def compute_fscore(dist1, dist2, thresholds=[0.005, 0.01, 0.02, 0.05, 0.1, 0.2]):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscores
    """
    fscores = []
    for threshold in thresholds:
        precision = torch.mean((dist1 < threshold).float(), dim=1)  # [B, ]
        recall = torch.mean((dist2 < threshold).float(), dim=1)
        fscore = 2 * precision * recall / (precision + recall)
        fscore[torch.isnan(fscore)] = 0
        fscores.append(fscore)
    fscores = torch.stack(fscores, dim=1)
    return fscores

def convert_to_explicit(opt, level_grids, isoval=0., to_pointcloud=False):
    N = len(level_grids)
    meshes = [None]*N
    pointclouds = [None]*N if to_pointcloud else None
    threads = [threading.Thread(target=convert_to_explicit_worker,
                                args=(opt, i, level_grids[i], isoval, meshes),
                                kwargs=dict(pointclouds=pointclouds),
                                daemon=False) for i in range(N)]
    for t in threads: t.start()
    for t in threads: t.join()
    if to_pointcloud:
        pointclouds = np.stack(pointclouds, axis=0)
        return meshes, pointclouds
    else: return meshes

def convert_to_explicit_worker(opt, i, level_vox_i, isoval, meshes, pointclouds=None):
    # use marching cubes to convert implicit surface to mesh
    vertices, faces = mcubes.marching_cubes(level_vox_i, isovalue=isoval)
    assert(level_vox_i.shape[0]==level_vox_i.shape[1]==level_vox_i.shape[2])
    S = level_vox_i.shape[0]
    range_min, range_max = opt.eval.range
    # marching cubes treat every cube as unit length
    vertices = vertices/S*(range_max-range_min)+range_min
    mesh = trimesh.Trimesh(vertices, faces)
    meshes[i] = mesh
    if pointclouds is not None:
        # randomly sample on mesh to get uniform dense point cloud
        if len(mesh.triangles)!=0:
            points = mesh.sample(opt.eval.num_points)
        else: points = np.zeros([opt.eval.num_points, 3])
        pointclouds[i] = points

def chamfer_distance(opt, X1, X2):
    B = len(X1)
    N1 = X1.shape[1]
    N2 = X2.shape[1]
    assert(X1.shape[2]==3)
    dist_1 = torch.zeros(B, N1, device=opt.device)
    dist_2 = torch.zeros(B, N2, device=opt.device)
    idx_1 = torch.zeros(B, N1, dtype=torch.int32, device=opt.device)
    idx_2 = torch.zeros(B, N2, dtype=torch.int32, device=opt.device)
    chamfer_3D.forward(X1, X2, dist_1, dist_2, idx_1, idx_2)
    return dist_1.sqrt(), dist_2.sqrt(), idx_1, idx_2