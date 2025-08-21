import os
import torch
import numpy as np
import random
import time
import logging
import logging.handlers
import open3d as o3d
from .gmm import GaussianMixture

THOUSAND = 1000
MILLION = 1000000


class BlackHole(object):
    def __setattr__(self, name, value):
        pass
    def __call__(self, *args, **kwargs):
        return self
    def __getattr__(self, name):
        return self


class CheckpointManager(object):

    def __init__(self, save_dir, logger=BlackHole()):
        super().__init__()
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self.ckpts = []
        self.logger = logger

        for f in os.listdir(self.save_dir):
            if f[:4] != 'ckpt':
                continue
            _, score, it = f.split('_')
            it = it.split('.')[0]
            self.ckpts.append({
                'score': float(score),
                'file': f,
                'iteration': int(it),
            })

    def get_worst_ckpt_idx(self):
        idx = -1
        worst = float('-inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] >= worst:
                idx = i
                worst = ckpt['score']
        return idx if idx >= 0 else None

    def get_best_ckpt_idx(self):
        idx = -1
        best = float('inf')
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['score'] <= best:
                idx = i
                best = ckpt['score']
        return idx if idx >= 0 else None
        
    def get_latest_ckpt_idx(self):
        idx = -1
        latest_it = -1
        for i, ckpt in enumerate(self.ckpts):
            if ckpt['iteration'] > latest_it:
                idx = i
                latest_it = ckpt['iteration']
        return idx if idx >= 0 else None

    def save(self, model, args, score, others=None, step=None):

        if step is None:
            fname = 'ckpt_%.6f_.pt' % float(score)
        else:
            fname = 'ckpt_%.6f_%d.pt' % (float(score), int(step))
        path = os.path.join(self.save_dir, fname)

        torch.save({
            'args': args,
            'state_dict': model.state_dict(),
            'others': others
        }, path)

        self.ckpts.append({
            'score': score,
            'file': fname
        })

        return True

    def load_best(self):
        idx = self.get_best_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt
    
    def load_latest(self):
        idx = self.get_latest_ckpt_idx()
        if idx is None:
            raise IOError('No checkpoints found.')
        ckpt = torch.load(os.path.join(self.save_dir, self.ckpts[idx]['file']))
        return ckpt

    def load_selected(self, file):
        ckpt = torch.load(os.path.join(self.save_dir, file))
        return ckpt


def seed_all(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_logger(name, log_dir=None):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('[%(asctime)s::%(name)s::%(levelname)s] %(message)s')

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.DEBUG)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    if log_dir is not None:
        file_handler = logging.FileHandler(os.path.join(log_dir, 'log.txt'))
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_new_log_dir(root='./logs', postfix='', prefix=''):
    log_dir = os.path.join(root, prefix + time.strftime('%Y_%m_%d__%H_%M_%S', time.localtime()) + postfix)
    os.makedirs(log_dir)
    return log_dir


def int_tuple(argstr):
    return tuple(map(int, argstr.split(',')))


def str_tuple(argstr):
    return tuple(argstr.split(','))


def int_list(argstr):
    return list(map(int, argstr.split(',')))


def str_list(argstr):
    return list(argstr.split(','))


def log_hyperparams(writer, log_dir, args):
    from torch.utils.tensorboard.summary import hparams
    vars_args = {k:v if isinstance(v, str) else repr(v) for k, v in vars(args).items()}
    exp, ssi, sei = hparams(vars_args, {"hp_metric": -1})
    fw = writer._get_file_writer()
    fw.add_summary(exp)
    fw.add_summary(ssi)
    fw.add_summary(sei)
    with open(os.path.join(log_dir, 'hparams.csv'), 'w') as csvf:
        csvf.write('key,value\n')
        for k, v in vars_args.items():
            csvf.write('%s,%s\n' % (k, v))



def get_data_iterator(iterable):
    """Allows training with DataLoaders in a single infinite loop:
        for i, data in enumerate(inf_generator(train_loader)):
    """
    iterator = iterable.__iter__()
    while True:
        try:
            yield iterator.__next__()
        except StopIteration:
            iterator = iterable.__iter__()


def parse_experiment_name(name):
    if 'blensor' in name:
        if 'Ours' in name:
            dataset, method, tag, blensor_, noise = name.split('_')[:5]
        else:
            dataset, method, blensor_, noise = name.split('_')[:4]
        return {
            'dataset': dataset,
            'method': method,
            'resolution': 'blensor',
            'noise': noise,
        }

    if 'real' in name:
        if 'Ours' in name:
            dataset, method, tag, blensor_, noise = name.split('_')[:5]
        else:
            dataset, method, blensor_, noise = name.split('_')[:4]
        return {
            'dataset': dataset,
            'method': method,
            'resolution': 'real',
            'noise': noise,
        }
        
    else:
        if 'Ours' in name:
            dataset, method, tag, num_pnts, sample_method, noise = name.split('_')[:6]
        else:
            dataset, method, num_pnts, sample_method, noise = name.split('_')[:5]
        return {
            'dataset': dataset,
            'method': method,
            'resolution': num_pnts + '_' + sample_method,
            'noise': noise,
        }

def addgaussiannoise(pts, mean):
    bbdiag = float(np.linalg.norm(pts.max(0) - pts.min(0), 2)) / 2
    noise = np.random.randn(pts.shape[0], pts.shape[1]) * bbdiag * mean
    out = pts + noise
    return out

def adddiffrentnoise(pts, noise_level, noise_class):
    bbdiag = float(np.linalg.norm(pts.max(0) - pts.min(0), 2)) / 2
    # 脉冲噪声
    if noise_class == 'pulse':
        pulse_intensity = bbdiag * noise_level * 3
        p = 0.2
        x = np.random.rand(pts.shape[0], pts.shape[1])
        f = np.zeros(shape=pts.shape)
        f[x < p / 2] = -pulse_intensity
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                if p / 2 < x[i, j] < p and True:
                    f[i, j] = pulse_intensity
        pulse_out = pts + f
        return pulse_out

    elif noise_class == 'uniform':
        b = bbdiag * noise_level * 3
        a = -bbdiag * noise_level * 3
        s_noise = a + (b - a) * np.random.rand(pts.shape[0], pts.shape[1])
        uniform_out = pts + s_noise
        return uniform_out

    elif noise_class == 'exp':
        a = bbdiag * noise_level
        e_noise = np.random.exponential(a, size=(pts.shape[0], pts.shape[1]))
        exp_out = pts + e_noise
        return exp_out

    elif noise_class == 'gaussian':
        bbdiag = float(np.linalg.norm(pts.max(0) - pts.min(0), 2)) / 2
        noise = np.random.randn(pts.shape[0], pts.shape[1]) * bbdiag * noise_level
        gauss_out = pts + noise
        return gauss_out

    else:
        print('No such noise')
        return None

def addmixnoise(pts, mean, noise_class, noise_level):
    bbdiag = float(np.linalg.norm(pts.max(0) - pts.min(0), 2)) / 2
    noise = np.random.randn(pts.shape[0], pts.shape[1]) * bbdiag * mean
    guass_out = pts + noise

    return adddiffrentnoise(guass_out, noise_level, noise_class)

def normal_estimate(pts_np):
    radius = 0.1  # 搜索半径
    max_nn = 30  # 邻域内用于估算法线的最大点数
    points_np = pts_np
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_np)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius, max_nn))
    normals = np.asarray(pcd.normals)
    # np.save('./Dataset/Test/boxunion2_100K_normal.npy', normals)
    return normals

# 输出两个点云点对点的差异向量
def point_diff(pts1, pts2):
    diff = pts2 - pts1
    return diff

# 点云加噪声，噪声为差异向量
def add_diff_noise(filtered_pt, noise_pts, level):
    diff = point_diff(filtered_pt, noise_pts)
    return filtered_pt + diff * level

def add_gmm_noise(filtered_pt, noise_pts, level):
    diff = point_diff(filtered_pt, noise_pts)
    # 保存差异向量为txt文件
    #np.savetxt("diff.txt", diff, fmt='%f', delimiter=' ')
    data = torch.tensor(diff, dtype=torch.float64)

    # Create an instance of the GaussianMixture class
    gmm = GaussianMixture(n_components=10, n_features=3)


    # Fit the model to the data
    gmm.fit(data)
    labels = gmm.predict(data)
    means, vars = gmm.get_means_vars()
    print(f'means: {means}, \n vars: {vars}')
    new_data, new_labels = gmm.sample(len(filtered_pt))
    new_data = new_data.numpy()
    return filtered_pt + new_data * level

def pgy(parameters1, parameters2, ratio=0.9):  # 参数平均
    "Used for convex summation"
    # new_params = []
    # for (params1, params2) in zip(parameters1, parameters2):
    #     ap = 9e-1
    #     gt = (1 - ap) * (params2) + (ap * params1)
    #     new_params.append((gt))

    average_dict = {}
    for key in parameters1:
        ap = ratio
        average_dict[key] = ap * parameters1[key].cpu() + (1-ap) * parameters2[key].cpu()
    return average_dict