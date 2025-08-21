import torch
import torch.nn as nn
import pytorch3d.ops
import numpy as np

from .feature import FeatureExtraction
from .score import ScoreNet


class LANScoreNet(nn.Module):
    """
    Learning to Adapt Noise (LAN) for Score-based Point Cloud Denoising
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        # geometry
        self.frame_knn = args.frame_knn
        self.num_train_points = args.num_train_points
        self.num_clean_nbs = args.num_clean_nbs
        if hasattr(args, 'num_selfsup_nbs'):
            self.num_selfsup_nbs = args.num_selfsup_nbs
        # score-matching
        self.dsm_sigma = args.dsm_sigma
        # networks
        self.feature_net = FeatureExtraction()
        self.score_net = ScoreNet(
            z_dim=self.feature_net.out_channels,
            dim=3, 
            out_dim=3,
            hidden_size=args.score_net_hidden_dim,
            num_blocks=args.score_net_num_blocks,
        )
        
        # Freeze the score network parameters if using LAN approach
        if hasattr(args, 'lan_mode') and args.lan_mode:
            for param in self.score_net.parameters():
                param.requires_grad = False
            for param in self.feature_net.parameters():
                param.requires_grad = False
                
    def get_supervised_loss(self, pcl_noisy, pcl_clean, lan=None):
        """
        Denoising score matching with LAN adaptation.
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
            pcl_clean:  Clean point clouds, (B, M, 3). Usually, M is slightly greater than N.
            lan: LAN module for noise adaptation
        """
        B, N_noisy, N_clean, d = pcl_noisy.size(0), pcl_noisy.size(1), pcl_clean.size(1), pcl_noisy.size(2)
        pnt_idx = np.random.permutation(N_noisy)[:self.num_train_points]
        
        # Apply LAN adaptation if provided
        if lan is not None:
            pcl_adapted = lan(pcl_noisy)
        else:
            pcl_adapted = pcl_noisy

        # Feature extraction
        feat = self.feature_net(pcl_adapted)  # (B, N, F)
        feat = feat[:,pnt_idx,:]  # (B, n, F)
        F = feat.size(-1)
        
        # Local frame construction
        _, _, frames = pytorch3d.ops.knn_points(pcl_adapted[:,pnt_idx,:], pcl_adapted, K=self.frame_knn, return_nn=True)  # (B, n, K, 3)
        frames_centered = frames - pcl_adapted[:,pnt_idx,:].unsqueeze(2)   # (B, n, K, 3)

        # Nearest clean points for each point in the local frame
        _, _, clean_nbs = pytorch3d.ops.knn_points(
            frames.view(-1, self.frame_knn, d),    # (B*n, K, 3)
            pcl_clean.unsqueeze(1).repeat(1, len(pnt_idx), 1, 1).view(-1, N_clean, d),   # (B*n, M, 3)
            K=self.num_clean_nbs,
            return_nn=True,
        )   # (B*n, K, C, 3)
        clean_nbs = clean_nbs.view(B, len(pnt_idx), self.frame_knn, self.num_clean_nbs, d)  # (B, n, K, C, 3)

        # Noise vectors
        noise_vecs = frames.unsqueeze(dim=3) - clean_nbs  # (B, n, K, C, 3)
        noise_vecs = noise_vecs.mean(dim=3)  # (B, n, K, 3)

        # Denoising score matching
        grad_pred = self.score_net(
            x = frames_centered.view(-1, self.frame_knn, d),
            c = feat.view(-1, F),
        ).reshape(B, len(pnt_idx), self.frame_knn, d)   # (B, n, K, 3)
        grad_target = - 1 * noise_vecs   # (B, n, K, 3)

        loss = 0.5 * ((grad_target - grad_pred) ** 2.0 * (1.0 / self.dsm_sigma)).sum(dim=-1).mean()
        
        return loss

    def get_selfsupervised_loss(self, pcl_noisy, lan=None):
        """
        Denoising score matching with LAN adaptation (self-supervised).
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
            lan: LAN module for noise adaptation
        """
        B, N_noisy, d = pcl_noisy.size()
        pnt_idx = np.random.permutation(N_noisy)[:self.num_train_points]
        
        # Apply LAN adaptation if provided
        if lan is not None:
            pcl_adapted = lan(pcl_noisy)
        else:
            pcl_adapted = pcl_noisy

        # Feature extraction
        feat = self.feature_net(pcl_adapted)  # (B, N, F)
        feat = feat[:,pnt_idx,:]  # (B, n, F)
        F = feat.size(-1)
        
        # Local frame construction
        _, _, frames = pytorch3d.ops.knn_points(pcl_adapted[:,pnt_idx,:], pcl_adapted, K=self.frame_knn, return_nn=True)  # (B, n, K, 3)
        frames_centered = frames - pcl_adapted[:,pnt_idx,:].unsqueeze(2)   # (B, n, K, 3)

        # Nearest points for each point in the local frame
        _, _, selfsup_nbs = pytorch3d.ops.knn_points(
            frames.view(-1, self.frame_knn, d),    # (B*n, K, 3)
            pcl_adapted.unsqueeze(1).repeat(1, len(pnt_idx), 1, 1).view(-1, N_noisy, d),   # (B*n, M, 3)
            K=self.num_selfsup_nbs,
            return_nn=True,
        )   # (B*n, K, C, 3)
        selfsup_nbs = selfsup_nbs.view(B, len(pnt_idx), self.frame_knn, self.num_selfsup_nbs, d)  # (B, n, K, C, 3)

        # Noise vectors
        noise_vecs = frames.unsqueeze(dim=3) - selfsup_nbs  # (B, n, K, C, 3)
        noise_vecs = noise_vecs.mean(dim=3)  # (B, n, K, 3)

        # Denoising score matching
        grad_pred = self.score_net(
            x = frames_centered.view(-1, self.frame_knn, d),
            c = feat.view(-1, F),
        ).reshape(B, len(pnt_idx), self.frame_knn, d)   # (B, n, K, 3)
        grad_target = - 1 * noise_vecs   # (B, n, K, 3)

        loss = 0.5 * ((grad_target - grad_pred) ** 2.0 * (1.0 / self.dsm_sigma)).sum(dim=-1).mean()
        return loss
  
    def denoise_langevin_dynamics(self, pcl_noisy, step_size, denoise_knn=4, step_decay=0.95, num_steps=30, lan=None):
        """
        Args:
            pcl_noisy:  Noisy point clouds, (B, N, 3).
            lan: LAN module for noise adaptation
        """
        B, N, d = pcl_noisy.size()
        with torch.no_grad():
            # Apply LAN adaptation if provided
            if lan is not None:
                pcl_adapted = lan(pcl_noisy)
            else:
                pcl_adapted = pcl_noisy
                
            # Feature extraction
            self.feature_net.eval()
            feat = self.feature_net(pcl_adapted)  # (B, N, F)
            _, _, F = feat.size()

            # Trajectories
            traj = [pcl_noisy.clone().cpu()]
            pcl_next = pcl_noisy.clone()

            for step in range(num_steps):
                # Apply LAN adaptation if provided
                if lan is not None:
                    pcl_adapted_next = lan(pcl_next)
                else:
                    pcl_adapted_next = pcl_next
                    
                # Construct local frames
                _, nn_idx, frames = pytorch3d.ops.knn_points(pcl_adapted, pcl_adapted_next, K=denoise_knn, return_nn=True)   
                frames_centered = frames - pcl_adapted.unsqueeze(2)   # (B, N, K, 3)
                nn_idx = nn_idx.view(B, -1)    # (B, N*K)

                # Predict gradients
                self.score_net.eval()
                grad_pred = self.score_net(
                    x=frames_centered.view(-1, denoise_knn, d),
                    c=feat.view(-1, F)
                ).reshape(B, -1, d)   # (B, N*K, 3)

                acc_grads = torch.zeros_like(pcl_noisy)
                acc_grads.scatter_add_(dim=1, index=nn_idx.unsqueeze(-1).expand_as(grad_pred), src=grad_pred)

                s = step_size * (step_decay ** step)
                pcl_next += s * acc_grads
                traj.append(pcl_next.clone().cpu())
            
        return pcl_next, traj


class LAN(nn.Module):
    """
    Learning to Adapt Noise (LAN) module for point cloud denoising
    """
    def __init__(self, shape=None):
        super(LAN, self).__init__()
        if shape is not None:
            self.phi = nn.Parameter(torch.zeros(shape), requires_grad=True)
        else:
            self.phi = None
            
    def set_shape(self, shape):
        """Initialize phi with the right shape if not done in __init__"""
        self.phi = nn.Parameter(torch.zeros(shape), requires_grad=True)
        
    def forward(self, x):
        """
        Add learnable noise offset to input point cloud
        Args:
            x: Input point cloud, (B, N, 3)
        Returns:
            Adapted point cloud, (B, N, 3)
        """
        if self.phi is None:
            self.set_shape(x.shape)
        return x + torch.tanh(self.phi)


def zsn2n_loss_func(noisy_pc, model, lan=None):
    """
    Zero-Shot Noise2Noise loss for point clouds
    Args:
        noisy_pc: Noisy point cloud, (B, N, 3)
        model: Score-based denoising model
        lan: LAN module
    """
    B, N, d = noisy_pc.size()
    
    # Apply LAN if provided
    if lan is not None:
        adapted_pc = lan(noisy_pc)
    else:
        adapted_pc = noisy_pc
    
    # Create two subsets of points by randomly sampling
    idx1 = torch.randperm(N, device=noisy_pc.device)[:N//2]
    idx2 = torch.randperm(N, device=noisy_pc.device)[:N//2]
    
    subset1 = adapted_pc[:, idx1, :]
    subset2 = adapted_pc[:, idx2, :]
    
    # Get denoised predictions
    with torch.no_grad():
        model.eval()
        denoised1, _ = model.denoise_langevin_dynamics(subset1, step_size=0.2, num_steps=10)
        denoised2, _ = model.denoise_langevin_dynamics(subset2, step_size=0.2, num_steps=10)
    
    # Cross-consistency loss
    loss_res = torch.nn.functional.mse_loss(subset1, denoised2) + torch.nn.functional.mse_loss(subset2, denoised1)
    
    # Self-consistency loss
    with torch.no_grad():
        denoised_full, _ = model.denoise_langevin_dynamics(adapted_pc, step_size=0.2, num_steps=10)
    
    denoised_full1 = denoised_full[:, idx1, :]
    denoised_full2 = denoised_full[:, idx2, :]
    
    loss_cons = torch.nn.functional.mse_loss(denoised1, denoised_full1) + torch.nn.functional.mse_loss(denoised2, denoised_full2)
    
    return loss_res + loss_cons


def nbr2nbr_loss_func(noisy_pc, model, lan=None):
    """
    Neighbor2Neighbor loss for point clouds
    Args:
        noisy_pc: Noisy point cloud, (B, N, 3)
        model: Score-based denoising model
        lan: LAN module
    """
    B, N, d = noisy_pc.size()
    
    # Apply LAN if provided
    if lan is not None:
        adapted_pc = lan(noisy_pc)
    else:
        adapted_pc = noisy_pc
    
    # For each point, find its k nearest neighbors
    k = 8  # Number of neighbors to consider
    _, _, neighbors = pytorch3d.ops.knn_points(adapted_pc, adapted_pc, K=k+1)
    neighbors = neighbors[:, :, 1:, :]  # Exclude the point itself (first neighbor)
    
    # Create two subsets by selecting different neighbors
    mask1 = torch.zeros(B, N, k, dtype=torch.bool, device=noisy_pc.device)
    mask2 = torch.zeros(B, N, k, dtype=torch.bool, device=noisy_pc.device)
    
    # Randomly assign neighbors to either mask1 or mask2
    for i in range(k):
        if i % 2 == 0:
            mask1[:, :, i] = True
        else:
            mask2[:, :, i] = True
    
    # Extract points based on masks
    subset1_idx = torch.randperm(N, device=noisy_pc.device)[:N//2]
    subset2_idx = torch.randperm(N, device=noisy_pc.device)[:N//2]
    
    subset1 = adapted_pc[:, subset1_idx, :]
    subset2 = adapted_pc[:, subset2_idx, :]
    
    # Get denoised predictions
    with torch.no_grad():
        model.eval()
        denoised = model.denoise_langevin_dynamics(adapted_pc, step_size=0.2, num_steps=10)[0]
    
    # Denoise the subsets
    denoised1 = model.denoise_langevin_dynamics(subset1, step_size=0.2, num_steps=10)[0]
    
    # Cross-consistency loss
    loss_nbr = torch.nn.functional.mse_loss(denoised1, subset2)
    
    # Self-consistency loss
    denoised_sub1 = denoised[:, subset1_idx, :]
    loss_cons = torch.nn.functional.mse_loss(denoised1, denoised_sub1)
    
    return loss_nbr + 0.1 * loss_cons 