import os
import torch
import numpy as np
from torch.utils.data import Dataset
from tqdm.auto import tqdm


class PointCloudDataset(Dataset):

    def __init__(self, root, dataset, split, resolution, transform=None, TTA=None):
        super().__init__()
        self.pcl_dir = os.path.join(root, dataset, 'pointclouds', split, resolution)
        self.transform = transform
        self.pointclouds = []
        self.pointcloud_names = []
        self.tta = TTA
        for fn in tqdm(os.listdir(self.pcl_dir), desc='Loading'):
            if fn[-3:] != 'xyz':
                continue
            pcl_path = os.path.join(self.pcl_dir, fn)
            if not os.path.exists(pcl_path):
                raise FileNotFoundError('File not found: %s' % pcl_path)
            pcl = torch.FloatTensor(np.loadtxt(pcl_path, dtype=np.float32))
            self.pointclouds.append(pcl)
            self.pointcloud_names.append(fn[:-4])

    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {
            'pcl_clean': self.pointclouds[idx].clone(), 
            'name': self.pointcloud_names[idx]
        }
        if self.transform is not None:
            data = self.transform(data)
        if self.tta is not None:
            data = {
                'pcl_clean': self.pointclouds[idx].clone(),
                'pcl_noisy1':self.tta[0],
                'pcl_noisy2':self.tta[1],
                'name': self.pointcloud_names[idx]
            }
        return data

class TTAPointCloudDataset(Dataset):

    def __init__(self, pointclouds, pointcloud_names, transform=None, TTA=None):
        super().__init__()
        self.pointclouds = []
        self.pointcloud_names = []
        self.transform = transform
        self.tta = TTA
        pcl = torch.FloatTensor(pointclouds)
        self.pointclouds.append(pcl)
        self.pointcloud_names.append(pointcloud_names)


    def __len__(self):
        return len(self.pointclouds)

    def __getitem__(self, idx):
        data = {
            'pcl_clean': self.pointclouds[idx].clone(),
            'name': self.pointcloud_names[idx]
        }
        if self.transform is not None:
            data = self.transform(data)
        if self.tta is not None:
            data = {
                'pcl_clean': self.pointclouds[idx].clone(),
                'pcl_noisy1': torch.FloatTensor(self.tta[0]),
                'pcl_noisy2': torch.FloatTensor(self.tta[1]),
                'name': self.pointcloud_names[idx]
            }
        return data