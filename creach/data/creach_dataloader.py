
# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#


import os
import glob
import random

import numpy as np
import torch
from torch.utils import data
from tools.utils import np2torch, torch2np
from tools.utils import to_cpu, to_np, to_tensor
from tools.cfg_parser import Config

from torch.utils.data.dataloader import default_collate
from omegaconf import DictConfig


import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

DEFAULT_NUM_WORKERS = {
    'train': 0,
    'val': 0,
    'test': 0
}

class LoadData(data.Dataset):
    def __init__(self,
                 cfg,
                 split_name='train'):

        super().__init__()

        self.split_name = split_name
        self.ds_dir = cfg.dataset_dir
        self.cfg = cfg
        dataset_dir = cfg.dataset_dir

        self.split_name = split_name
        self.ds_dir = dataset_dir

        self.ds = {}
        self.ds_path = os.path.join(dataset_dir,split_name)
        datasets = glob.glob(self.ds_path + '/*.npy')

        self.load_ds(datasets)
        self.frame_names = np.load(os.path.join(dataset_dir,split_name, 'frame_names.npz'))['frame_names']
        self.frame_st_end = np.asarray([int(name.split('_')[-1]) for name in self.frame_names])
        self.ds.pop('dataset', None)

    def load_ds(self, dataset_names):
        self.ds = {}
        for name in dataset_names:
            self.ds.update(np.load(name, allow_pickle=True))
        self.ds = np2torch(self.ds)

    def normalize(self):

        norm_data_dir = os.path.join(self.ds_dir,'norm_data.pt')
        if os.path.exists(norm_data_dir):
            self.norm_data = torch.load(norm_data_dir)
        elif self.split_name =='train':
            in_p = {k: (v.mean(0, keepdim=True), v.std(0, keepdim=True) + 1e-10) for k, v in self.ds['in'].items() if v.dtype==torch.float}
            out_p = {k: (v.mean(0, keepdim=True), v.std(0, keepdim=True) + 1e-10) for k, v in self.ds['out'].items()}
            self.norm_data = {'in':in_p, 'out':out_p}
            torch.save(self.norm_data,norm_data_dir)
        else:
            raise('Please run the train split first to normalize the data')

        in_p = self.norm_data['in']
        out_p = self.norm_data['out']

        for k, v in in_p.items():
            self.ds['in'][k] = (self.ds['in'][k]-v[0])/v[1]


    def load_idx(self, idx, source=None):

        if source is None:
            source = self.ds

        out = {}
        for k, v in source.items():
            if isinstance(v, dict):
                out[k] = self.load_idx(idx, v)
            else:
                out[k] = v[idx]
        out['dataset'] = 1 if 'circle/' in self.frame_names[idx] else 0

        return out

    def __len__(self):
        return self.ds['fullpose'].shape[0]

    def __getitem__(self, idx):
        
        data_out = self.load_idx(idx)
        data_out['idx'] = torch.from_numpy(np.array(idx, dtype=np.int32))
        return data_out

def loc2vel(loc,fps):
    B = loc.shape[0]
    idxs = [0] + list(range(B-1))
    vel = (loc[1:] - loc[:-1])/(1/float(fps))
    return vel[idxs]


def build_dataloader(dataset: torch.utils.data.Dataset,
                     cfg: DictConfig,
                     split: str = 'train',
                     ) -> torch.utils.data.DataLoader:

    dataset_cfg = cfg
    is_train    = 'train' in split
    is_test    = 'test' in split
    num_workers = dataset_cfg.get('num_workers', DEFAULT_NUM_WORKERS)
    shuffle     = dataset_cfg.get('shuffle', True)

    collate_fn  = None

    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size  =   dataset_cfg.batch_size if not is_test else 1,
        num_workers =   num_workers.get(split, 0),
        collate_fn  =   collate_fn,
        drop_last   =   True and (is_train or not is_test),
        pin_memory  =   dataset_cfg.get('pin_memory', False),
        shuffle     =   shuffle and is_train and not is_test,
    )
    return data_loader


