import os
import argparse
import torch
import torch.utils.tensorboard
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm.auto import tqdm

from datasets import *
from utils.misc import *
from utils.transforms import *
from utils.denoise import *
from models.lan_score import *
from models.utils import chamfer_distance_unit_sphere


# Arguments
parser = argparse.ArgumentParser()
## Dataset and loader
parser.add_argument('--dataset_root', type=str, default='./data')
parser.add_argument('--dataset', type=str, default='PUNet')
parser.add_argument('--patch_size', type=int, default=1000)
parser.add_argument('--resolutions', type=str_list, default=['10000_poisson', '30000_poisson', '50000_poisson'])
parser.add_argument('--noise_min', type=float, default=0.005)
parser.add_argument('--noise_max', type=float, default=0.020)
parser.add_argument('--train_batch_size', type=int, default=32)
parser.add_argument('--num_workers', type=int, default=4)
parser.add_argument('--aug_rotate', type=eval, default=True, choices=[True, False])
## Model architecture
parser.add_argument('--supervised', type=eval, default=True, choices=[True, False])
parser.add_argument('--frame_knn', type=int, default=32)
parser.add_argument('--num_train_points', type=int, default=128)
parser.add_argument('--num_clean_nbs', type=int, default=4, help='For supervised training.')
parser.add_argument('--num_selfsup_nbs', type=int, default=8, help='For self-supervised training.')
parser.add_argument('--dsm_sigma', type=float, default=0.01)
parser.add_argument('--score_net_hidden_dim', type=int, default=128)
parser.add_argument('--score_net_num_blocks', type=int, default=4)
## LAN specific arguments
parser.add_argument('--lan_mode', type=eval, default=True, choices=[True, False], help='Whether to use LAN adaptation')
parser.add_argument('--pretrained_model', type=str, default='./pretrained/ckpt.pt', help='Path to pretrained score model')
parser.add_argument('--self_loss', type=str, default='zsn2n', choices=['zsn2n', 'nbr2nbr'], help='Self-supervised loss function')
parser.add_argument('--inner_loop', type=int, default=20, help='Number of iterations for inner optimization loop')
## Optimizer and scheduler
parser.add_argument('--lr', type=float, default=1e-4)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--max_grad_norm', type=float, default=float("inf"))
## Training
parser.add_argument('--seed', type=int, default=2020)
parser.add_argument('--logging', type=eval, default=True, choices=[True, False])
parser.add_argument('--log_root', type=str, default='./logs')
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--max_iters', type=int, default=1*MILLION)
parser.add_argument('--val_freq', type=int, default=2000)
parser.add_argument('--val_upsample_rate', type=int, default=4)
parser.add_argument('--val_num_visualize', type=int, default=4)
parser.add_argument('--val_noise', type=float, default=0.015)
parser.add_argument('--ld_step_size', type=float, default=0.2)
parser.add_argument('--tag', type=str, default=None)
args = parser.parse_args()
seed_all(args.seed)

# Logging
if args.logging:
    log_dir = get_new_log_dir(args.log_root, prefix='LAN_D%s_' % (args.dataset), postfix='_' + args.tag if args.tag is not None else '')
    logger = get_logger('train', log_dir)
    writer = torch.utils.tensorboard.SummaryWriter(log_dir)
    ckpt_mgr = CheckpointManager(log_dir)
    log_hyperparams(writer, log_dir, args)
else:
    logger = get_logger('train', None)
    writer = BlackHole()
    ckpt_mgr = BlackHole()
logger.info(args)

# Datasets and loaders
logger.info('Loading datasets')
train_dset = PairedPatchDataset(
    datasets=[
        PointCloudDataset(
            root=args.dataset_root,
            dataset=args.dataset,
            split='train',
            resolution=resl,
            transform=standard_train_transforms(noise_std_max=args.noise_max, noise_std_min=args.noise_min, rotate=args.aug_rotate)
        ) for resl in args.resolutions
    ],
    patch_size=args.patch_size,
    patch_ratio=1.2,
    on_the_fly=True  
)
val_dset = PointCloudDataset(
        root=args.dataset_root,
        dataset=args.dataset,
        split='test',
        resolution=args.resolutions[0],
        transform=standard_train_transforms(noise_std_max=args.val_noise, noise_std_min=args.val_noise, rotate=False, scale_d=0),
    )
train_iter = get_data_iterator(DataLoader(train_dset, batch_size=args.train_batch_size, num_workers=args.num_workers, shuffle=True))

# Load pretrained model
logger.info('Loading pretrained model...')
if os.path.exists(args.pretrained_model):
    ckpt = torch.load(args.pretrained_model, map_location=args.device)
    pretrained_args = ckpt['args']
    # Update necessary args from pretrained model
    args.frame_knn = pretrained_args.frame_knn
    args.num_train_points = pretrained_args.num_train_points
    args.num_clean_nbs = pretrained_args.num_clean_nbs
    if hasattr(pretrained_args, 'num_selfsup_nbs'):
        args.num_selfsup_nbs = pretrained_args.num_selfsup_nbs
    args.dsm_sigma = pretrained_args.dsm_sigma
    args.score_net_hidden_dim = pretrained_args.score_net_hidden_dim
    args.score_net_num_blocks = pretrained_args.score_net_num_blocks
else:
    logger.warning(f"Pretrained model not found at {args.pretrained_model}. Training from scratch.")
    ckpt = None

# Model
logger.info('Building model...')
model = LANScoreNet(args).to(args.device)
if ckpt is not None:
    # Load pretrained weights
    model.load_state_dict(ckpt['state_dict'], strict=False)
    logger.info("Loaded pretrained model weights")
logger.info(repr(model))

# Setup loss function based on self_loss argument
if args.self_loss == 'zsn2n':
    loss_func = zsn2n_loss_func
elif args.self_loss == 'nbr2nbr':
    loss_func = nbr2nbr_loss_func
else:
    raise NotImplementedError(f"Self-supervised loss {args.self_loss} not implemented")

# Train, validate and test
def train_lan_batch(model, pcl_noisy, pcl_clean):
    """
    Train LAN on a single batch using the specified self-supervised loss
    """
    # Create LAN module for this batch
    lan = LAN(pcl_noisy.shape).to(args.device)
    
    # Setup optimizer for LAN
    optimizer = torch.optim.Adam(lan.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # Metrics tracking
    logs = {'psnr': [], 'cd': []}
    
    # Initial metrics before adaptation
    with torch.no_grad():
        model.eval()
        # Get initial denoised result without LAN
        pcl_denoised_init, _ = model.denoise_langevin_dynamics(pcl_noisy, step_size=args.ld_step_size)
        # Calculate initial metrics
        init_psnr = calculate_psnr(pcl_denoised_init, pcl_clean)
        init_cd = chamfer_distance_unit_sphere(pcl_denoised_init.unsqueeze(0), pcl_clean.unsqueeze(0))[0].item()
        logs['psnr'].append(init_psnr)
        logs['cd'].append(init_cd)
    
    # Inner optimization loop
    for i in range(args.inner_loop):
        # Reset gradients
        optimizer.zero_grad()
        
        # Calculate loss
        loss = loss_func(pcl_noisy, model, lan)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Evaluate current performance
        if (i + 1) % 5 == 0 or i == args.inner_loop - 1:
            with torch.no_grad():
                model.eval()
                # Apply LAN adaptation
                pcl_adapted = lan(pcl_noisy)
                # Get denoised result with LAN
                pcl_denoised, _ = model.denoise_langevin_dynamics(pcl_noisy, step_size=args.ld_step_size, lan=lan)
                # Calculate metrics
                curr_psnr = calculate_psnr(pcl_denoised, pcl_clean)
                curr_cd = chamfer_distance_unit_sphere(pcl_denoised.unsqueeze(0), pcl_clean.unsqueeze(0))[0].item()
                logs['psnr'].append(curr_psnr)
                logs['cd'].append(curr_cd)
    
    return logs, lan

def calculate_psnr(pred, target, max_val=1.0):
    """Calculate PSNR between point clouds"""
    # Use MSE as a proxy for PSNR in point clouds
    with torch.no_grad():
        mse = torch.mean((pred - target) ** 2)
        psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()

def train(it):
    # Load data
    batch = next(train_iter)
    pcl_noisy = batch['pcl_noisy'].to(args.device)
    pcl_clean = batch['pcl_clean'].to(args.device)

    # Reset model state
    model.eval()  # Keep model in eval mode since we're only training LAN

    # Initialize metrics
    batch_logs = {'psnr': [], 'cd': []}
    
    # Process each example in the batch
    for i in range(pcl_noisy.size(0)):
        # Train LAN on this example
        logs, _ = train_lan_batch(model, pcl_noisy[i], pcl_clean[i])
        
        # Collect metrics
        for key in logs:
            if i == 0:
                batch_logs[key] = logs[key]
            else:
                for j in range(len(logs[key])):
                    batch_logs[key][j] += logs[key][j]
    
    # Average metrics across batch
    for key in batch_logs:
        batch_logs[key] = [val / pcl_noisy.size(0) for val in batch_logs[key]]

    # Logging
    logger.info('[Train] Iter %04d | PSNR %.2f->%.2f | CD %.6f->%.6f' % (
        it, 
        batch_logs['psnr'][0], batch_logs['psnr'][-1],
        batch_logs['cd'][0], batch_logs['cd'][-1]
    ))
    writer.add_scalar('train/psnr_init', batch_logs['psnr'][0], it)
    writer.add_scalar('train/psnr_final', batch_logs['psnr'][-1], it)
    writer.add_scalar('train/cd_init', batch_logs['cd'][0], it)
    writer.add_scalar('train/cd_final', batch_logs['cd'][-1], it)
    writer.add_scalar('train/psnr_improvement', batch_logs['psnr'][-1] - batch_logs['psnr'][0], it)
    writer.add_scalar('train/cd_improvement', batch_logs['cd'][0] - batch_logs['cd'][-1], it)
    writer.flush()

def validate(it):
    all_clean = []
    all_denoised_init = []
    all_denoised_lan = []
    
    for i, data in enumerate(tqdm(val_dset, desc='Validate')):
        if i >= args.val_num_visualize:
            break
            
        pcl_noisy = data['pcl_noisy'].to(args.device)
        pcl_clean = data['pcl_clean'].to(args.device)
        
        # Denoise without LAN
        pcl_denoised_init = patch_based_denoise(model, pcl_noisy, ld_step_size=args.ld_step_size)
        
        # Train LAN for this example
        logs, lan = train_lan_batch(model, pcl_noisy, pcl_clean)
        
        # Denoise with LAN
        pcl_denoised_lan = patch_based_denoise(model, pcl_noisy, ld_step_size=args.ld_step_size, lan=lan)
        
        all_clean.append(pcl_clean.unsqueeze(0))
        all_denoised_init.append(pcl_denoised_init.unsqueeze(0))
        all_denoised_lan.append(pcl_denoised_lan.unsqueeze(0))
    
    all_clean = torch.cat(all_clean, dim=0)
    all_denoised_init = torch.cat(all_denoised_init, dim=0)
    all_denoised_lan = torch.cat(all_denoised_lan, dim=0)

    # Calculate metrics
    cd_init = chamfer_distance_unit_sphere(all_denoised_init, all_clean, batch_reduction='mean')[0].item()
    cd_lan = chamfer_distance_unit_sphere(all_denoised_lan, all_clean, batch_reduction='mean')[0].item()

    logger.info('[Val] Iter %04d | CD Init %.6f | CD LAN %.6f | Improvement %.6f' % (
        it, cd_init, cd_lan, cd_init - cd_lan
    ))
    writer.add_scalar('val/cd_init', cd_init, it)
    writer.add_scalar('val/cd_lan', cd_lan, it)
    writer.add_scalar('val/cd_improvement', cd_init - cd_lan, it)
    writer.add_mesh('val/pcl_init', all_denoised_init, global_step=it)
    writer.add_mesh('val/pcl_lan', all_denoised_lan, global_step=it)
    writer.flush()

    return cd_lan

# Main loop
logger.info('Start training...')
try:
    for it in range(1, args.max_iters+1):
        train(it)
        if it % args.val_freq == 0 or it == args.max_iters:
            cd_loss = validate(it)
            # Save checkpoint
            ckpt_mgr.save(model, args, cd_loss, {}, step=it)

except KeyboardInterrupt:
    logger.info('Terminating...') 