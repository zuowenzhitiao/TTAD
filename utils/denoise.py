import math
import torch
import numpy as np
import pytorch3d.ops
from tqdm.auto import tqdm
from sklearn.cluster import KMeans
from sklearn.neighbors import kneighbors_graph, KDTree

from models.utils import farthest_point_sampling
from .transforms import NormalizeUnitSphere


def patch_based_denoise(model, pcl_noisy, ld_step_size=0.2, ld_num_steps=30, patch_size=1000, seed_k=3, denoise_knn=4, step_decay=0.95, get_traj=False, lan=None):
    """
    Args:
        pcl_noisy:  Input point cloud, (N, 3)
        lan: LAN module for noise adaptation (optional)
    """
    assert pcl_noisy.dim() == 2, 'The shape of input point cloud must be (N, 3).'
    N, d = pcl_noisy.size()
    pcl_noisy = pcl_noisy.unsqueeze(0)  # (1, N, 3)
    seed_pnts, _ = farthest_point_sampling(pcl_noisy, int(seed_k * N / patch_size))
    _, _, patches = pytorch3d.ops.knn_points(seed_pnts, pcl_noisy, K=patch_size, return_nn=True)
    patches = patches[0]    # (N, K, 3)

    with torch.no_grad():
        model.eval()
        patches_denoised, traj = model.denoise_langevin_dynamics(patches, step_size=ld_step_size, denoise_knn=denoise_knn, step_decay=step_decay, num_steps=ld_num_steps, lan=lan)

    pcl_denoised, fps_idx = farthest_point_sampling(patches_denoised.view(1, -1, d), N)
    pcl_denoised = pcl_denoised[0]
    fps_idx = fps_idx[0]

    if get_traj:
        for i in range(len(traj)):
            traj[i] = traj[i].view(-1, d)[fps_idx, :]
        return pcl_denoised, traj
    else:
        return pcl_denoised


def denoise_large_pointcloud(model, pcl, cluster_size, seed=0, lan=None):
    """
    Denoise large point cloud by clustering and processing each cluster separately
    
    Args:
        model: Denoising model
        pcl: Input point cloud
        cluster_size: Maximum number of points per cluster
        seed: Random seed for KMeans
        lan: LAN module for noise adaptation (optional)
    """
    device = pcl.device
    pcl = pcl.cpu().numpy()

    print('Running KMeans to construct clusters...')
    n_clusters = math.ceil(pcl.shape[0] / cluster_size)
    kmeans = KMeans(n_clusters=n_clusters, random_state=seed).fit(pcl)

    pcl_parts = []
    for i in tqdm(range(n_clusters), desc='Denoise Clusters'):
        pts_idx = kmeans.labels_ == i

        pcl_part_noisy = torch.FloatTensor(pcl[pts_idx]).to(device)
        pcl_part_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_part_noisy)
        
        # Create LAN module for this cluster if needed
        cluster_lan = None
        if lan is not None:
            from models.lan_score import LAN
            cluster_lan = LAN(pcl_part_noisy.unsqueeze(0).shape).to(device)
            # Copy parameters from the original LAN if possible
            if hasattr(lan, 'phi') and lan.phi is not None:
                # Resize phi to match the current cluster size
                with torch.no_grad():
                    # Simple approach: just use the first N values from the original phi
                    # A more sophisticated approach would be to interpolate or use nearest neighbors
                    N = min(cluster_lan.phi.shape[1], lan.phi.shape[1])
                    cluster_lan.phi.data[:, :N, :] = lan.phi.data[:, :N, :]
        
        pcl_part_denoised = patch_based_denoise(
            model,
            pcl_part_noisy,
            seed_k=5,
            lan=cluster_lan
        )
        pcl_part_denoised = pcl_part_denoised * scale + center
        pcl_parts.append(pcl_part_denoised)

    return torch.cat(pcl_parts, dim=0)
