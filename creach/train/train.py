# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Universiteit van Amsterdam (UvA).
# All rights reserved.
#
# Universiteit van Amsterdam (UvA) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with UvA or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: g.paschalidis@uva.nl
#


import os
import shutil
import sys
sys.path.append('.')
sys.path.append('..')
import json
import numpy as np
import torch
import smplx
import argparse
from smplx import SMPLXLayer, SMPLX


from datetime import datetime
from tools.train_tools import EarlyStopping


from torch import nn, optim

from tensorboardX import SummaryWriter

import glob, time


from tools.utils import makepath, makelogger, to_cpu, to_np, to_tensor
from loguru import logger

from tools.utils import aa2rotmat, rotmat2aa, rotmul, rotate

from omegaconf import OmegaConf

from models.cvae import CReach
from losses import build_loss
from optimizers import build_optimizer
from data.creach_dataloader import LoadData, build_dataloader

from tools.utils import aa2rotmat, rotmat2aa, d62rotmat
from models.model_utils import parms_6D2full
from tools.train_tools import v2v
from tqdm import tqdm

from tools.utils import LOGGER_DEFAULT_FORMAT
from configs.creach_config import conf as cfg


class Trainer:

    def __init__(self,cfg, inference=False):

        self.dtype = torch.float32
        self.cfg = cfg
        self.is_inference = inference
        
        self.joint_start_rh = self.cfg.joint_start_rh
        self.joint_target_rh = self.cfg.joint_target_rh

        self.joint_start_lh = self.cfg.joint_start_lh
        self.joint_target_lh = self.cfg.joint_target_lh
        
        torch.manual_seed(cfg.seed)

        starttime = datetime.now().replace(microsecond=0)
        makepath(cfg.work_dir, isfile=False)
        logger_path = makepath(os.path.join(cfg.work_dir, '%s_%s.log' % (cfg.expr_ID, 'train' if not inference else 'test')), isfile=True)

        logger.add(logger_path,  backtrace=True, diagnose=True)
        logger.add(lambda x:x,
                   level=cfg.logger_level.upper(),
                   colorize=True,
                   format=LOGGER_DEFAULT_FORMAT
                   )
        self.logger = logger.info

        summary_logdir = os.path.join(cfg.work_dir, 'summaries')
        self.swriter = SummaryWriter(log_dir=summary_logdir)
        self.logger('[%s] - Started training XXX, experiment code %s' % (cfg.expr_ID, starttime))
        self.logger('tensorboard --logdir=%s' % summary_logdir)
        self.logger('Torch Version: %s\n' % torch.__version__)

        stime = datetime.now().replace(microsecond=0)
        shutil.copy2(sys.argv[0], os.path.join(cfg.work_dir, os.path.basename(sys.argv[0]).replace('.py', '_%s.py' % datetime.strftime(stime, '%Y%m%d_%H%M'))))

        use_cuda = torch.cuda.is_available()
        if use_cuda:
            torch.cuda.empty_cache()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        gpu_brand = torch.cuda.get_device_name(cfg.cuda_id) if use_cuda else None
        gpu_count = cfg.num_gpus
        if use_cuda:
            self.logger('Using %d CUDA cores [%s] for training!' % (gpu_count, gpu_brand))

        self.data_info = {}
        self.load_data(cfg, inference)


        self.body_model_cfg = cfg.body_model

        self.predict_offsets = cfg.get('predict_offsets', False)
        self.logger(f'Predict offsets: {self.predict_offsets}')

        self.use_exp = cfg.get('use_exp', 0)
        self.logger(f'Use exp function on distances: {self.use_exp}')

        model_path = os.path.join(self.body_model_cfg.get('model_path', 'data/models'), 'smplx')

        self.body_model = SMPLXLayer(
            model_path=model_path,
            gender='neutral',
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(self.device)

        self.female_model = SMPLXLayer(
            model_path=model_path,
            gender='female',
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(self.device)

        self.male_model = SMPLXLayer(
            model_path=model_path,
            gender='male',
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(self.device)

        self.network = CReach(**cfg.network.creach).to(self.device)

        # Setup the training losses
        self.loss_setup()

        if cfg.num_gpus > 1:
            self.network = nn.DataParallel(self.network)
            self.logger("Training on Multiple GPU's")

        vars_network = [var[1] for var in self.network.named_parameters()]
        n_params = sum(p.numel() for p in vars_network if p.requires_grad)
        self.logger('Total Trainable Parameters for network is %2.2f M.' % ((n_params) * 1e-6))

        self.configure_optimizers()

        self.best_loss = np.inf

        self.epochs_completed = 0
        self.cfg = cfg
        self.network.cfg = cfg

        if inference and cfg.best_model is None:
            cfg.best_model = sorted(glob.glob(os.path.join(cfg.work_dir, 'snapshots', '*[0-9][0-9][0-9]_model.pt')))[-1]
        if cfg.best_model is not None:
            self._get_network().load_state_dict(torch.load(cfg.best_model, map_location=self.device), strict=False)
            self.logger('Restored trained model from %s' % cfg.best_model)
            self.epochs_completed = int((cfg.best_model.split("/")[-1]).split("_")[0][1:]) + 1


    def loss_setup(self):

        self.logger('Configuring the losses!')

        loss_cfg = self.cfg.get('losses', {})

        self.LossL1 = nn.L1Loss(reduction='mean')
        self.LossL2 = nn.MSELoss(reduction='mean')
        self.Lossbce = nn.BCELoss(reduction='mean')

        # Edge loss
        edge_loss_cfg = loss_cfg.get('edge', {})
        self.edge_loss = build_loss(**edge_loss_cfg)
        self.edge_loss_weight = edge_loss_cfg.get('weight', 0.0)
        self.logger(f'Edge loss, weight: {self.edge_loss}, {self.edge_loss_weight}')

        # Vertex loss
        # TODO: Add denser vertex sampling
        vertex_loss_cfg = loss_cfg.get('vertices', {})
        self.vertex_loss_weight = vertex_loss_cfg.get('weight', 0.0)
        self.vertex_loss = build_loss(**vertex_loss_cfg)
        self.logger(f'Vertex loss, weight: {self.vertex_loss},'
                    f' {self.vertex_loss_weight}')

        z_min_loss_cfg = loss_cfg.get('z_min', {})
        self.z_min_loss_weight = z_min_loss_cfg.get('weight', 0.0)
        self.z_min_loss = build_loss(**z_min_loss_cfg)
        self.logger(f'z min loss, weight: {self.z_min_loss},'
                    f' {self.z_min_loss_weight}')
        

        vertex_consist_loss_cfg = loss_cfg.get('vertices_consist', {})
        self.vertex_consist_loss_weight = vertex_consist_loss_cfg.get('weight', 0.0)
        # self.vertex_loss = build_loss(**vertex_loss_cfg)
        self.logger(f'Vertex consist loss weight: {self.vertex_consist_loss_weight}')

        rh_vertex_loss_cfg = loss_cfg.get('rh_vertices', {})
        self.rh_vertex_loss_weight = rh_vertex_loss_cfg.get('weight', 10.0)
        self.rh_vertex_loss = build_loss(**rh_vertex_loss_cfg)
        self.logger(f'Right Hand Vertex loss, weight: {self.rh_vertex_loss},'
                     f' {self.rh_vertex_loss_weight}')

        lh_vertex_loss_cfg = loss_cfg.get('lh_vertices', {})
        self.lh_vertex_loss_weight = lh_vertex_loss_cfg.get('weight', 10.0)
        self.lh_vertex_loss = build_loss(**lh_vertex_loss_cfg)
        self.logger(f'Left Hand Vertex loss, weight: {self.lh_vertex_loss},'
                     f' {self.lh_vertex_loss_weight}')
        
        joint_loss_cfg = loss_cfg.get('start_joint', {})
        self.joint_loss_weight = joint_loss_cfg.get('weight', 10.0)
        self.joint_loss = build_loss(**joint_loss_cfg)
        self.logger(f'Start joint loss, weight: {self.joint_loss},'
                     f' {self.joint_loss_weight}')
       
        feet_vertex_loss_cfg = loss_cfg.get('feet_vertices', {})
        self.feet_vertex_loss_weight = feet_vertex_loss_cfg.get('weight', 0.0)
        self.feet_vertex_loss = build_loss(**feet_vertex_loss_cfg)
        self.logger(f'Feet Vertex loss, weight: {self.feet_vertex_loss},'
                     f' {self.feet_vertex_loss_weight}')

        pose_loss_cfg = loss_cfg.get('pose', {})
        self.pose_loss_weight = pose_loss_cfg.get('weight', 0.0)
        self.pose_loss = build_loss(**pose_loss_cfg)
        self.logger(f'Pose loss, weight: {self.pose_loss},'
                    f' {self.pose_loss}')

        velocity_loss_cfg = loss_cfg.get('velocity', {})
        self.velocity_loss_weight = velocity_loss_cfg.get('weight', 0.0)
        self.velocity_loss = build_loss(**velocity_loss_cfg)

        self.logger(f'Velocity loss, weight: {self.velocity_loss},'
                    f' {self.velocity_loss_weight}')

        acceleration_loss_cfg = loss_cfg.get('acceleration', {})
        self.acceleration_loss_weight = acceleration_loss_cfg.get('weight', 0.0)
        self.acceleration_loss = build_loss(**acceleration_loss_cfg)
        self.logger(
            f'Acceleration loss, weight: {self.acceleration_loss},'
            f' {self.acceleration_loss_weight}')

        contact_loss_cfg = loss_cfg.get('contact', {})
        self.contact_loss_weight = contact_loss_cfg.get('weight', 0.0)
        self.logger(
            f'Contact loss, weight: '
            f' {self.contact_loss_weight}')

        kl_loss_cfg = loss_cfg.get('kl_loss', {})
        self.kl_loss_weight = kl_loss_cfg.get('weight', 0.0)
        self.logger(
            f'KL loss, weight: '
            f' {self.kl_loss_weight}')

        self.verts_ids = to_tensor(np.load(self.cfg.datasets.verts_sampled), dtype=torch.long)
        self.rhand_idx = torch.from_numpy(np.load(loss_cfg.rh2smplx_idx))
        self.lhand_idx = torch.from_numpy(np.load(loss_cfg.lh2smplx_idx))
        self.rh_ids_sampled = torch.tensor(np.where([id in self.rhand_idx for id in self.verts_ids])[0]).to(torch.long)

        self.feet_idx = torch.from_numpy(np.load(loss_cfg.feet2smplx_idx))


    def load_data(self,cfg, inference):
        self.logger('Base dataset_dir is %s' % self.cfg.datasets.dataset_dir)

        ds_name = 'test'
        self.data_info[ds_name] = {}
        ds_test = LoadData(self.cfg.datasets, split_name=ds_name)
        self.data_info[ds_name]['frame_names'] = ds_test.frame_names
        self.ds_test = build_dataloader(ds_test, split='test', cfg=self.cfg.datasets)

        if not inference:

            ds_name = 'train'
            self.data_info[ds_name] = {}
            ds_train = LoadData(self.cfg.datasets, split_name=ds_name)
            self.data_info[ds_name]['frame_names'] = ds_train.frame_names
            self.ds_train = build_dataloader(ds_train, split=ds_name, cfg=self.cfg.datasets)

            ds_name = 'val'
            self.data_info[ds_name] = {}
            ds_val = LoadData(self.cfg.datasets, split_name=ds_name)
            self.data_info[ds_name]['frame_names'] = ds_val.frame_names
            self.ds_val = build_dataloader(ds_val, split=ds_name, cfg=self.cfg.datasets)

        if not inference:
            self.logger('Dataset Train, Val, Test size respectively: %.2f M, %.2f K, %.2f K' %
                        (len(self.ds_train.dataset) * 1e-6, len(self.ds_val.dataset) * 1e-3, len(self.ds_test.dataset) * 1e-3))

    def edges_for(self, x, vpe):
        return (x[:, vpe[:, 0]] - x[:, vpe[:, 1]])

    def _get_network(self):
        return self.network.module if isinstance(self.network, torch.nn.DataParallel) else self.network

    def save_network(self):
        torch.save(self.network.module.state_dict()
                   if isinstance(self.network, torch.nn.DataParallel)
                   else self.network.state_dict(), self.cfg.best_model)

    def forward(self, x):
        if self.is_inference:
            return self.infer(x)
        
        bs = x['transl'].shape[0]
        rh_grasp_mask = x["grasp_type"] == 0
        lh_grasp_mask = ~rh_grasp_mask
        dec_x = {}
        enc_x = {}

        enc_x['fullpose'] = x['fullpose_rotmat'][:,:,:2,:]
        wrist_transl = torch.zeros((bs,3)).to(self.device)
        wrist_transl[rh_grasp_mask] = x['joints'][rh_grasp_mask, self.joint_start_rh].clone()
        wrist_transl[lh_grasp_mask] = x['joints'][lh_grasp_mask, self.joint_start_lh].clone()
        
        enc_x['transl'] = x['transl'] - wrist_transl
        obj_loc = torch.zeros(bs).to(self.device)
        obj_loc[rh_grasp_mask] = x['joints'][rh_grasp_mask, self.joint_start_rh, 1]
        obj_loc[lh_grasp_mask] = x['joints'][lh_grasp_mask, self.joint_start_lh, 1]

        enc_x['obj_loc'] = obj_loc
        dec_x['obj_loc'] = obj_loc

        direction = torch.zeros((bs,3)).to(self.device)
        direction[rh_grasp_mask] = x['joints'][rh_grasp_mask, self.joint_target_rh,:] - x['joints'][rh_grasp_mask, self.joint_start_rh,:]
        direction[lh_grasp_mask] = x['joints'][lh_grasp_mask, self.joint_target_lh,:] - x['joints'][lh_grasp_mask, self.joint_start_lh,:]
        norm_dir = direction / ((direction**2).sum(1).sqrt()[...,None] + 1e-8)
        dec_x["dir"] = norm_dir
        enc_x["dir"] = norm_dir
        
        dec_x['betas'] = x['betas']
        enc_x['betas'] = x['betas']
        
        enc_x = torch.cat([v.reshape(bs, -1).to(self.device) for v in enc_x.values()], dim=1)
        z_enc, mean, var = self.network.encode(enc_x)
        z_enc_s = z_enc.rsample()
        dec_x['z'] = z_enc_s
        dec_x["grasp_type"] = x["grasp_type"].to(torch.float32)
        
        dec_x = torch.cat([v.reshape(bs, -1).to(self.device) for v in dec_x.values()], dim=1)
        
        net_output = self.network.decode(dec_x)

        pose, trans = net_output['pose'], net_output['trans'] + wrist_transl
        creach_output = self.prepare_rnet(x, pose, trans)
        results = {}
        results['z_enc'] = {'mean': mean, 'std': var}
        creach_output.update(net_output)
        results['creach'] = creach_output
    
        return  results

    def prepare_rnet(self, batch, pose, trans):

        d62rot = pose.shape[-1] == 330
        bparams = parms_6D2full(pose, trans, d62rot=d62rot)
        
        grasp_type = batch["grasp_type"]

        genders = batch['gender']
        males = genders == 1
        females = ~males

        B, _, _ = batch['joints'].shape
        v_template = batch['betas'].to(self.device)

        FN = sum(females)
        MN = sum(males)
        f_refnet_params = {}
        m_refnet_params = {}
        creach_output = {}
        refnet_in = {}

        if FN > 0:
            f_params = {k: v[females] for k, v in bparams.items()}
            f_params['betas'] = v_template[females]
            f_params["grasp_type"] = grasp_type[females]
            rh_mask = f_params["grasp_type"] == 0
            lh_mask = ~rh_mask
            f_output = self.female_model(**f_params)
            f_verts = f_output.vertices
            f_joints = f_output.joints
            f_dir = torch.zeros((len(f_joints),3)).to(self.device)
            f_dir[rh_mask] = f_joints[rh_mask,self.joint_target_rh,:] - f_joints[rh_mask,self.joint_start_rh,:]
            f_dir[lh_mask] = f_joints[lh_mask,self.joint_target_lh,:] - f_joints[lh_mask,self.joint_start_lh,:]
            f_dir_norm = f_dir / ((f_dir**2).sum(1).sqrt()[...,None] + 1e-8)
            
 
            creach_output['f_verts_full'] = f_verts
            creach_output['f_params'] = f_params
            creach_output["f_dir_norm"] = f_dir_norm
         

        if MN > 0:

            m_params = {k: v[males] for k, v in bparams.items()}
            m_params['betas'] = v_template[males]
            m_params["grasp_type"] = grasp_type[males]
            rh_mask = m_params["grasp_type"] == 0
            lh_mask = ~rh_mask
            m_output = self.male_model(**m_params)
            m_verts = m_output.vertices
               
            m_joints = m_output.joints
            
            m_dir = torch.zeros((len(m_joints),3)).to(self.device)
            m_dir[rh_mask] = m_joints[rh_mask,self.joint_target_rh,:] - m_joints[rh_mask,self.joint_start_rh,:]
            m_dir[lh_mask] = m_joints[lh_mask,self.joint_target_lh,:] - m_joints[lh_mask,self.joint_start_lh,:]
            m_dir_norm = m_dir / ((m_dir**2).sum(1).sqrt()[...,None] + 1e-8)
            
            creach_output['m_verts_full'] = m_verts
            creach_output['m_params'] = m_params
            creach_output["m_dir_norm"] = m_dir_norm
            
        return creach_output


    def train(self):
        self.network.train()
        save_every_it = len(self.ds_train) / self.cfg.summary_steps
        
        train_loss_dict = {}
        
        for it, batch in tqdm(enumerate(self.ds_train)):
            batch = {k: batch[k].to(self.device) for k in batch.keys()}

            self.optimizer.zero_grad()
            output = self.forward(batch)
            loss_total, losses_dict = self.get_loss(batch, output)

            loss_total.backward()

            self.optimizer.step()

            train_loss_dict = {k: train_loss_dict.get(k, 0.0) + v.item() for k, v in losses_dict.items()}
            if it % (save_every_it + 1) == 0:
                cur_train_loss_dict = {k: v / (it + 1) for k, v in train_loss_dict.items()}
                train_msg = self.create_loss_message(cur_train_loss_dict,
                                                    expr_ID=self.cfg.expr_ID,
                                                    epoch_num=self.epochs_completed,
                                                    model_name='creach',
                                                    it=it,
                                                    try_num=0,
                                                    mode='train')

                self.logger(train_msg)

            

        train_loss_dict = {k: v / len(self.ds_train) for k, v in train_loss_dict.items()}
        
        return train_loss_dict

    def evaluate(self, ds_name='val'):
        self.network.eval()

        eval_loss_dict = {}

        data = self.ds_val if ds_name == 'val' else self.ds_test
        with torch.no_grad():
            for it, batch in enumerate(data):

                batch = {k: batch[k].to(self.device) for k in batch.keys()}

                self.optimizer.zero_grad()

                output = self.forward(batch)

                loss_total, losses_dict = self.get_loss(batch, output)

                eval_loss_dict = {k: eval_loss_dict.get(k, 0.0) + v for k, v in losses_dict.items()}

            eval_loss_dict = {k: v / len(data) for k, v in eval_loss_dict.items()}

        return eval_loss_dict

    def get_loss(self, batch, results):
        enc_z = results['z_enc']
        creach = results['creach']
        genders = batch['gender']
        males = genders == 1
        females = ~males
        
        B, _, _= batch['joints'].shape
        v_template = batch['betas'].to(self.device)
        grasp_type  = batch['grasp_type'].to(self.device)

        FN = sum(females)
        MN = sum(males)

        params_gt = parms_6D2full(batch['fullpose_rotmat'],
                                  batch['transl'],
                                  d62rot=False)
        if FN > 0:

            f_params_gt = {k: v[females] for k, v in params_gt.items()}
            f_params_gt['betas'] = v_template[females]
            f_params_gt['grasp_type'] = grasp_type[females]
            f_rh_grasp = f_params_gt["grasp_type"] == 0
            f_lh_grasp = ~f_rh_grasp
            f_output_gt = self.female_model(**f_params_gt)
            f_verts_gt = f_output_gt.vertices

        if MN > 0:

            m_params_gt = {k: v[males] for k, v in params_gt.items()}
            m_params_gt['betas'] = v_template[males]
            m_params_gt['grasp_type'] = grasp_type[males]
            m_rh_grasp = m_params_gt["grasp_type"] == 0
            m_lh_grasp = ~m_rh_grasp
            m_output_gt = self.male_model(**m_params_gt)
            m_verts_gt = m_output_gt.vertices

        losses = {}
        
        if self.vertex_loss_weight > 0:
            losses['creach_vertices'] = 0
            if FN > 0:
                losses['creach_vertices'] += self.vertex_loss(f_verts_gt, creach['f_verts_full'])
            if MN > 0:
                losses['creach_vertices'] += self.vertex_loss(m_verts_gt, creach['m_verts_full'])
            losses['creach_vertices'] *= self.vertex_loss_weight
        else:
            losses["creach_vertices"] = torch.zeros(1).to(self.device)
            

        if self.pose_loss_weight > 0:
            losses['creach_pose'] = 0
            losses['creach_trans'] = 0
            if FN>0:
                losses['creach_pose'] += self.LossL2(f_params_gt['fullpose_rotmat'], creach['f_params']['fullpose_rotmat'])
                losses['creach_trans'] += self.LossL1(f_params_gt['transl'], creach['f_params']['transl'])
            if MN>0:
                losses['creach_pose'] += self.LossL2(m_params_gt['fullpose_rotmat'], creach['m_params']['fullpose_rotmat'])
                losses['creach_trans'] += self.LossL1(m_params_gt['transl'], creach['m_params']['transl'])

            losses['creach_pose'] *= self.pose_loss_weight
            losses['creach_trans'] *= self.pose_loss_weight

        else:
            losses["creach_pose"] = torch.zeros(1).to(self.device)
            losses["creach_trans"] = torch.zeros(1).to(self.device)


        # right hand vertex loss
        if self.rh_vertex_loss_weight > 0:
            losses['creach_rh_vertices'] = torch.tensor(0).to(torch.float32).to(self.device)
            if FN > 0 and f_rh_grasp.sum()>0:
                losses['creach_rh_vertices'] += self.rh_vertex_loss(f_verts_gt[f_rh_grasp][:, self.rhand_idx], creach['f_verts_full'][f_rh_grasp][:, self.rhand_idx])
            if MN > 0 and m_rh_grasp.sum()>0:
                losses['creach_rh_vertices'] += self.rh_vertex_loss(m_verts_gt[m_rh_grasp][:, self.rhand_idx], creach['m_verts_full'][m_rh_grasp][:, self.rhand_idx])

            losses['creach_rh_vertices'] *= self.rh_vertex_loss_weight
        else:
            losses["creach_rh_vertices"] = torch.tensor(0).to(torch.float32).to(self.device)
        
        # left hand vertex loss
        if self.lh_vertex_loss_weight > 0:
            losses['creach_lh_vertices'] = torch.tensor(0).to(torch.float32).to(self.device)
            if FN > 0 and f_lh_grasp.sum()>0:
                losses['creach_lh_vertices'] += self.lh_vertex_loss(f_verts_gt[f_lh_grasp][:, self.lhand_idx], creach['f_verts_full'][f_lh_grasp][:, self.lhand_idx])
            if MN > 0 and m_lh_grasp.sum()>0:
                losses['creach_lh_vertices'] += self.lh_vertex_loss(m_verts_gt[m_lh_grasp][:, self.lhand_idx], creach['m_verts_full'][m_lh_grasp][:, self.lhand_idx])

            losses['creach_lh_vertices'] *= self.lh_vertex_loss_weight
        else:
            losses["creach_lh_vertices"] = torch.tensor(0).to(torch.float32).to(self.device)
        

        losses["direction"] = 0

        if FN >0:
            f_dir_gt = torch.zeros((FN,3)).to(self.device)
            f_dir_gt[f_rh_grasp] = batch["joints"][females][f_rh_grasp,self.joint_target_rh] - batch["joints"][females][f_rh_grasp,self.joint_start_rh]
            f_dir_gt[f_lh_grasp] = batch["joints"][females][f_lh_grasp,self.joint_target_lh] - batch["joints"][females][f_lh_grasp,self.joint_start_lh]
            norm_f_dir_gt = f_dir_gt / ((f_dir_gt**2).sum(1).sqrt()[...,None] + 1e-8)
            losses['direction'] += self.LossL1(norm_f_dir_gt, creach['f_dir_norm'])

        if MN > 0:
            m_dir_gt = torch.zeros((MN,3)).to(self.device)
            m_dir_gt[m_rh_grasp] = batch["joints"][males][m_rh_grasp,self.joint_target_rh] - batch["joints"][males][m_rh_grasp,self.joint_start_rh]
            m_dir_gt[m_lh_grasp] = batch["joints"][males][m_lh_grasp,self.joint_target_lh] - batch["joints"][males][m_lh_grasp,self.joint_start_lh]
            norm_m_dir_gt = m_dir_gt / ((m_dir_gt**2).sum(1).sqrt()[...,None] + 1e-8)
            losses['direction'] += self.LossL1(norm_m_dir_gt, creach['m_dir_norm'])

        losses["direction"] *= 5
        
        e_z = torch.distributions.normal.Normal(enc_z['mean'], enc_z['std'])  # encoder distribution

        n_z = torch.distributions.normal.Normal(
            loc=torch.zeros([self.cfg.datasets.batch_size, self.cfg.network.creach.latentD], requires_grad=False).to(self.device).type(self.dtype),
            scale=torch.ones([self.cfg.datasets.batch_size, self.cfg.network.creach.latentD], requires_grad=False).to(self.device).type(self.dtype))
        
        # kl between the encoder and normal distribution
        losses['loss_kl_encoder']   = self.kl_loss_weight * torch.mean(torch.sum(torch.distributions.kl.kl_divergence(e_z, n_z), dim=[1]))  

        with torch.no_grad():
            loss_v2v = []

            if FN > 0:
                loss_v2v.append(v2v(f_verts_gt,
                                     creach['f_verts_full'],
                                     mean=False)
                                 )
            if MN > 0:
                loss_v2v.append(v2v(m_verts_gt,
                                     creach['m_verts_full'],
                                     mean=False)
                                 )

            loss_v2v = torch.cat(loss_v2v, dim=0).mean(dim=-1).sum()
        loss_total = torch.stack(list(losses.values())).sum()
        losses['loss_total'] = loss_total
        losses['loss_v2v'] = loss_v2v
        return loss_total, losses

    def set_loss_weights(self):

        if self.epochs_completed > 3:
            self.vertex_loss_weight = 8
        else:
            self.vertex_loss_weight = 5

        if self.epochs_completed >= 5:
            self.rh_vertex_loss_weight = 15
            self.lh_vertex_loss_weight = 15


    def fit(self, n_epochs=None, message=None):
        starttime = datetime.now().replace(microsecond=0)
        if n_epochs is None:
            n_epochs = self.cfg.n_epochs

        self.logger('Started Training at %s for %d epochs' % (datetime.strftime(starttime, '%Y-%m-%d_%H:%M:%S'), n_epochs))
        if message is not None:
            self.logger(message)

        prev_lr = np.inf
        for epoch_num in range(self.epochs_completed, n_epochs + 1):
            self.logger('--- starting Epoch # %03d' % epoch_num)
            train_loss_dict = self.train()
            eval_loss_dict  = self.evaluate()

            self.set_loss_weights()


            self.lr_scheduler.step(eval_loss_dict['loss_v2v'])
            cur_lr = self.optimizer.param_groups[0]['lr']

            if cur_lr != prev_lr:
                self.logger('--- learning rate changed from %.2e to %.2e ---' % (prev_lr, cur_lr))
                prev_lr = cur_lr

            with torch.no_grad():
                eval_msg = Trainer.create_loss_message(eval_loss_dict, expr_ID=self.cfg.expr_ID,
                                                        epoch_num=self.epochs_completed, it=len(self.ds_val),
                                                        model_name='creach',
                                                        try_num=0, mode='evald')
                if eval_loss_dict['loss_v2v'] < self.best_loss:

                    self.cfg.best_model = makepath(os.path.join(self.cfg.work_dir, 'snapshots', 'E%03d_model.pt' % (self.epochs_completed)), isfile=True)
                    self.save_network()
                    self.logger(eval_msg + ' ** ')
                    self.best_loss = eval_loss_dict['loss_v2v']

                else:
                    self.logger(eval_msg)

                self.swriter.add_scalars('total_loss/scalars',
                                         {'train_loss_total': train_loss_dict['loss_total'],
                                         'evald_loss_total': eval_loss_dict['loss_total'], },
                                         self.epochs_completed)

            if self.early_stopping(eval_loss_dict['loss_v2v']):
                self.logger('Early stopping the training!')
                break

            self.epochs_completed += 1

        endtime = datetime.now().replace(microsecond=0)

        self.logger('Finished Training at %s\n' % (datetime.strftime(endtime, '%Y-%m-%d_%H:%M:%S')))
        self.logger('Training done in %s! Best val total loss achieved: %.2e\n' % (endtime - starttime, self.best_loss))
        self.logger('Best model path: %s\n' % self.cfg.best_model)

    def configure_optimizers(self):
        self.optimizer = build_optimizer([self.network], self.cfg.optim)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', factor=.2, patience=8)
        self.early_stopping = EarlyStopping(**self.cfg.network.early_stopping, trace_func=self.logger)

    @staticmethod
    def create_loss_message(loss_dict, expr_ID='XX', epoch_num=0,model_name='mlp', it=0, try_num=0, mode='evald'):
        ext_msg = ' | '.join(['%s = %.2e' % (k, v) for k, v in loss_dict.items() if k != 'loss_total'])
        return '[%s]_TR%02d_E%03d - It %05d - %s - %s: [T:%.2e] - [%s]' % (
            expr_ID, try_num, epoch_num, it,model_name, mode, loss_dict['loss_total'], ext_msg)

def train():

    parser = argparse.ArgumentParser(description='CReach-Training')

    parser.add_argument('--work-dir',
                        required=True,
                        type=str,
                        help='The path to the folder to save results')

    parser.add_argument('--grab-circle-path',
                        required=True,
                        type=str,
                        help='The path to the folder that contains GRAB data')

    parser.add_argument('--smplx-path',
                        required=True,
                        type=str,
                        help='The path to the folder containing SMPL-X model downloaded from the website')

    parser.add_argument('--expr-id', default='CReach_V00', type=str,
                        help='Training ID')

    parser.add_argument('--batch-size', default=32, type=int,
                        help='Training batch size')

    parser.add_argument('--num-gpus', default=1,
                        type=int,
                        help='Number of multiple GPUs for training')

    cmd_args = parser.parse_args()


    cfg.expr_ID = cfg.expr_ID if cmd_args.expr_id is None else cmd_args.expr_id

    cfg.datasets.dataset_dir = os.path.join(cmd_args.grab_circle_path,'CReach_data')
    cfg.datasets.grab_path = cmd_args.grab_circle_path
    cfg.body_model.model_path = cmd_args.smplx_path

    cfg.output_folder = cmd_args.work_dir
    cfg.results_base_dir = os.path.join(cfg.output_folder, 'results')
    cfg.num_gpus = cmd_args.num_gpus

    cfg.work_dir = os.path.join(cfg.output_folder, cfg.expr_ID)
    makepath(cfg.work_dir)
    ########################################
    run_trainer_once(cfg)

def run_trainer_once(cfg):

    trainer = Trainer(cfg=cfg)
    OmegaConf.save(trainer.cfg, os.path.join(cfg.work_dir, '{}.yaml'.format(cfg.expr_ID)))

    trainer.fit()

    OmegaConf.save(trainer.cfg, os.path.join(cfg.work_dir, '{}.yaml'.format(cfg.expr_ID)))



if __name__ == '__main__':

    train()
