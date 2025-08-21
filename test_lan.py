import os
import time
import argparse
import torch
import numpy as np
from tqdm.auto import tqdm

from utils.misc import *
from utils.transforms import *
from utils.denoise import *
from utils.evaluate import *
from models.lan_score import *

def input_iter(input_dir):
    for fn in os.listdir(input_dir):
        if fn[-3:] != 'xyz':
            continue
        pcl_noisy = torch.FloatTensor(np.loadtxt(os.path.join(input_dir, fn)))
        pcl_noisy, center, scale = NormalizeUnitSphere.normalize(pcl_noisy)
        yield {
            'pcl_noisy': pcl_noisy,
            'name': fn[:-4],
            'center': center,
            'scale': scale
        }

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, default='./pretrained/ckpt.pt')
parser.add_argument('--input_root', type=str, default='./data/examples')
parser.add_argument('--output_root', type=str, default='./data/results')
parser.add_argument('--dataset_root', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='PUNet')
parser.add_argument('--tag', type=str, default='')
parser.add_argument('--resolution', type=str, default='10000_poisson')
parser.add_argument('--noise', type=str, default='0.02')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--seed', type=int, default=2020)
# Denoiser parameters
parser.add_argument('--ld_step_size', type=float, default=None)
parser.add_argument('--ld_step_decay', type=float, default=0.95)
parser.add_argument('--ld_num_steps', type=int, default=30)
parser.add_argument('--seed_k', type=int, default=3)
parser.add_argument('--niters', type=int, default=1)
parser.add_argument('--denoise_knn', type=int, default=4, help='Number of score functions to be ensembled')
# LAN parameters
parser.add_argument('--use_lan', type=eval, default=True, choices=[True, False], help='Whether to use LAN adaptation')
parser.add_argument('--self_loss', type=str, default='zsn2n', choices=['zsn2n', 'nbr2nbr'], help='Self-supervised loss function')
parser.add_argument('--inner_loop', type=int, default=20, help='Number of iterations for inner optimization loop')
parser.add_argument('--lan_lr', type=float, default=1e-4, help='Learning rate for LAN optimization')
args = parser.parse_args()
seed_all(args.seed)

# Input/Output
input_dir = os.path.join(args.input_root, '%s_%s_%s' % (args.dataset, args.resolution, args.noise))
save_title = '{dataset}_LAN{modeltag}_{tag}_{res}_{noise}_{time}'.format_map({
    'dataset': args.dataset,
    'modeltag': '' if args.niters == 1 else '%dx' % args.niters,
    'tag': args.tag,
    'res': args.resolution,
    'noise': args.noise,
    'time': time.strftime('%m-%d-%H-%M-%S', time.localtime())
})
output_dir = os.path.join(args.output_root, save_title)
os.makedirs(output_dir, exist_ok=True)
os.makedirs(os.path.join(output_dir, 'pcl'), exist_ok=True)    # Output point clouds
logger = get_logger('test', output_dir)
for k, v in vars(args).items():
    logger.info('[ARGS::%s] %s' % (k, repr(v)))

# Model
ckpt = torch.load(args.ckpt, map_location=args.device)
model = LANScoreNet(ckpt['args']).to(args.device)
model.load_state_dict(ckpt['state_dict'], strict=False)
model.eval()

# Setup loss function based on self_loss argument
if args.self_loss == 'zsn2n':
    loss_func = zsn2n_loss_func
elif args.self_loss == 'nbr2nbr':
    loss_func = nbr2nbr_loss_func
else:
    raise NotImplementedError(f"Self-supervised loss {args.self_loss} not implemented")

# Denoise
ld_step_size = args.ld_step_size if args.ld_step_size is not None else ckpt['args'].ld_step_size
logger.info('ld_step_size = %.8f' % ld_step_size)

for data in input_iter(input_dir):
    logger.info(data['name'])
    pcl_noisy = data['pcl_noisy'].to(args.device)
    
    # Create LAN module for this point cloud if needed
    lan = None
    if args.use_lan:
        lan = LAN(pcl_noisy.unsqueeze(0).shape).to(args.device)
        optimizer = torch.optim.Adam(lan.parameters(), lr=args.lan_lr)
        
        # Optimize LAN parameters
        logger.info("Optimizing LAN parameters...")
        for i in tqdm(range(args.inner_loop)):
            optimizer.zero_grad()
            loss = loss_func(pcl_noisy.unsqueeze(0), model, lan)
            loss.backward()
            optimizer.step()
        
        # Log adapted point cloud
        with torch.no_grad():
            pcl_adapted = lan(pcl_noisy.unsqueeze(0))[0]
        
        # Save adapted point cloud
        pcl_adapted_denorm = pcl_adapted * data['scale'] + data['center']
        save_path = os.path.join(output_dir, 'pcl', data['name'] + '_adapted.xyz')
        np.savetxt(save_path, pcl_adapted_denorm.cpu().numpy(), fmt='%.8f')
    
    with torch.no_grad():
        model.eval()
        pcl_next = pcl_noisy
        for _ in range(args.niters):
            pcl_next = patch_based_denoise(
                model=model,
                pcl_noisy=pcl_next,
                ld_step_size=ld_step_size,
                ld_num_steps=args.ld_num_steps,
                step_decay=args.ld_step_decay,
                seed_k=args.seed_k,
                denoise_knn=args.denoise_knn,
                lan=lan
            )
        pcl_denoised = pcl_next.cpu()
        # Denormalize
        pcl_denoised = pcl_denoised * data['scale'] + data['center']
    
    save_path = os.path.join(output_dir, 'pcl', data['name'] + '.xyz')
    np.savetxt(save_path, pcl_denoised.numpy(), fmt='%.8f')

# Evaluate
evaluator = Evaluator(
    output_pcl_dir=os.path.join(output_dir, 'pcl'),
    dataset_root=args.dataset_root,
    dataset=args.dataset,
    summary_dir=args.output_root,
    experiment_name=save_title,
    device=args.device,
    res_gts=args.resolution,
    logger=logger
)
evaluator.run()


def test_large_pointcloud():
    """Test function for large point clouds"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_xyz', type=str, default='./data/examples/large.xyz')
    parser.add_argument('--output_xyz', type=str, default='./data/results/large_denoised.xyz')
    parser.add_argument('--ckpt', type=str, default='./pretrained/ckpt.pt')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--cluster_size', type=int, default=20000)
    parser.add_argument('--use_lan', type=eval, default=True, choices=[True, False])
    parser.add_argument('--self_loss', type=str, default='zsn2n', choices=['zsn2n', 'nbr2nbr'])
    parser.add_argument('--inner_loop', type=int, default=20)
    parser.add_argument('--lan_lr', type=float, default=1e-4)
    args_large = parser.parse_args()

    # Load model
    ckpt = torch.load(args_large.ckpt, map_location=args_large.device)
    model = LANScoreNet(ckpt['args']).to(args_large.device)
    model.load_state_dict(ckpt['state_dict'], strict=False)
    model.eval()

    # Setup loss function
    if args_large.self_loss == 'zsn2n':
        loss_func_large = zsn2n_loss_func
    elif args_large.self_loss == 'nbr2nbr':
        loss_func_large = nbr2nbr_loss_func

    # Load point cloud
    if os.path.exists(args_large.input_xyz):
        pcl = torch.FloatTensor(np.loadtxt(args_large.input_xyz)).to(args_large.device)
    else:
        # Example point cloud
        pcl = torch.randn(50000, 3).to(args_large.device)
        pcl = pcl / torch.norm(pcl, dim=1, keepdim=True) + 0.1 * torch.randn_like(pcl)
    
    # Normalize
    pcl, center, scale = NormalizeUnitSphere.normalize(pcl)
    
    # Create LAN module
    lan = None
    if args_large.use_lan:
        # For large point clouds, we'll optimize a smaller LAN on a subset
        # and then use it for the full point cloud
        subset_size = min(10000, pcl.size(0))
        subset_idx = torch.randperm(pcl.size(0))[:subset_size]
        pcl_subset = pcl[subset_idx].unsqueeze(0)
        
        lan_subset = LAN(pcl_subset.shape).to(args_large.device)
        optimizer = torch.optim.Adam(lan_subset.parameters(), lr=args_large.lan_lr)
        
        # Optimize LAN parameters on subset
        print("Optimizing LAN parameters on subset...")
        for i in tqdm(range(args_large.inner_loop)):
            optimizer.zero_grad()
            loss = loss_func_large(pcl_subset, model, lan_subset)
            loss.backward()
            optimizer.step()
        
        # Create full LAN
        lan = LAN(pcl.unsqueeze(0).shape).to(args_large.device)
        # Copy parameters from subset LAN
        with torch.no_grad():
            # Expand phi to match the full point cloud size
            lan.phi.data[0, :subset_size, :] = lan_subset.phi.data[0]
    
    # Denoise
    print('Denoising large point cloud...')
    pcl_denoised = denoise_large_pointcloud(model, pcl, args_large.cluster_size, lan=lan)
    
    # Denormalize
    pcl_denoised = pcl_denoised * scale + center
    
    # Save result
    print('Saving result to', args_large.output_xyz)
    os.makedirs(os.path.dirname(args_large.output_xyz), exist_ok=True)
    np.savetxt(args_large.output_xyz, pcl_denoised.cpu().numpy(), fmt='%.8f')


if __name__ == "__main__":
    # If called directly, run test_large_pointcloud
    test_large_pointcloud() 