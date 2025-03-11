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
import sys
import os
import numpy as np
import torch
import glob, json
import smplx
import argparse
import shutil
import time
from datetime import datetime
from tqdm import tqdm
from torch.utils import data

from tools.utils import makepath, makelogger
from tools.utils import parse_npz, parse_npz_circle
from tools.utils import params2torch
from tools.utils import prepare_params
from tools.utils import to_cpu, to_tensor
from tools.utils import append2dict
from tools.utils import torch2np
from tools.utils import aa2rotmat, rotmat2aa, rotate, rotmul

from bps_torch.bps import bps_torch

from bps_torch.tools import sample_sphere_uniform
from bps_torch.tools import sample_uniform_cylinder

from creach.test.vis_utils import read_o3d_mesh

import random
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INTENTS = ['lift', 'pass', 'offhand', 'use', 'all']


class CReachDataset(object):

    def __init__(self, cfg, logger=None, **params):

        self.cfg = cfg
        self.grab_path = cfg.grab_path
        self.circle_path = cfg.circle_path
        self.save_path = cfg.save_path
        makepath(self.save_path)


        if logger is None:
            log_dir = os.path.join(self.save_path, 'grab_preprocessing.log')
            self.logger = makelogger(log_dir=log_dir, mode='a').info
        else:
            self.logger = logger
        self.logger('Starting data preprocessing !')

        # assert cfg.intent in INTENTS

        self.intent = cfg.intent
        self.logger('intent:%s --> processing %s sequences!' % (self.intent, self.intent))

        if cfg.splits is None:
            self.splits = { 'test': .1,
                            'val': .05,
                            'train': .85}
        else:
            assert isinstance(cfg.splits, dict)
            self.splits = cfg.splits
        
        self.splits_prob = {'test': .1,
                       'val': .05,
                       'train': .85}
        self.all_seqs = glob.glob(self.grab_path + '/*/*.npz')
        self.all_seqs_circle = glob.glob(self.circle_path + '/*/*/*/*/*_reaching.npz')
        random.shuffle(self.all_seqs_circle)

        num_sequences_per_split = {split: int(len(self.all_seqs_circle) * probability) for split, probability in self.splits_prob.items()}

        ## Split the sequences into the train, validation and test sets.
        train_set = self.all_seqs_circle[:num_sequences_per_split['train']]
        val_set = self.all_seqs_circle[num_sequences_per_split['train']:num_sequences_per_split['train'] + num_sequences_per_split['val']]
        test_set = self.all_seqs_circle[num_sequences_per_split['train'] + num_sequences_per_split['val']:num_sequences_per_split['train'] + num_sequences_per_split['val'] + num_sequences_per_split['test']]

        self.split_seqs_circle = {  'train': train_set,
                                    'val': val_set,
                                    'test': test_set}

        
        ### to be filled 
        self.selected_seqs = []
        self.obj_based_seqs = {}
        self.sbj_based_seqs = {}
        self.split_seqs = {'test': [],
                           'val': [],
                           'train': []
                           }

        ### group, mask, and sort sequences based on objects, subjects, and intents
        self.process_sequences()

        self.logger('Total sequences: %d' % len(self.all_seqs))
        self.logger('Selected sequences: %d' % len(self.selected_seqs))
        self.logger('Number of sequences in each data split : train: %d , test: %d , val: %d'
                         %(len(self.split_seqs['train']), len(self.split_seqs['test']), len(self.split_seqs['val'])))
        ### process the data
        self.data_preprocessing(cfg)


    def data_preprocessing(self,cfg):

        self.obj_info = {}
        self.sbj_info = {}

        bps_fname = makepath(os.path.join(cfg.save_path, 'bps.pt'), isfile=True)

        self.bps_torch = bps_torch()

        R_bps = torch.tensor(
            [[1., 0., 0.],
             [0., 0., -1.],
             [0., 1., 0.]]).reshape(1, 3, 3).to(device)

        bps_path = 'configs/bps.pt'

        if os.path.exists(bps_path):
            self.bps = torch.load(bps_path)
            self.logger(f'loading bps from {bps_path}')
        else:
            self.bps_obj = sample_sphere_uniform(n_points=cfg.n_obj, radius=cfg.r_obj).reshape(1, -1, 3)
            self.bps_sbj = rotate(sample_uniform_cylinder(n_points=cfg.n_sbj, radius=cfg.r_sbj, height=cfg.h_sbj).reshape(1, -1, 3), R_bps.transpose(1, 2))
            self.bps_rh = sample_sphere_uniform(n_points=cfg.n_rh, radius=cfg.r_rh).reshape(1, -1, 3)
            self.bps_hd = sample_sphere_uniform(n_points=cfg.n_hd, radius=cfg.r_hd).reshape(1, -1, 3)

            self.bps = {
                'obj':self.bps_obj.cpu(),
                'sbj':self.bps_sbj.cpu(),
                'rh':self.bps_rh.cpu(),
                'hd':self.bps_hd.cpu(),
            }
            torch.save(self.bps,bps_fname)

        verts_ids = to_tensor(np.load('consts/verts_ids_0512.npy'), dtype=torch.long)
        rh_verts_ids = to_tensor(np.load('consts/rhand_smplx_ids.npy'), dtype=torch.long)

        stime = datetime.now().replace(microsecond=0)
        shutil.copy2(sys.argv[0],
                     os.path.join(self.save_path,
                                  os.path.basename(sys.argv[0]).replace('.py','_%s.py' % datetime.strftime(stime,'%Y%m%d_%H%M'))))

        self.subject_mesh = {}
        self.obj_info = {}
        self.sbj_info = {}

        for split in self.split_seqs.keys():
            outfname = makepath(os.path.join(cfg.save_path, split, 'grasp_motion_data.npy'), isfile=True)

            if os.path.exists(outfname):
                self.logger('Results for %s split already exist.' % (split))
                continue
            else:
                self.logger('Processing data for %s split.' % (split))


            frame_names = []
            n_frames = -1

            grasp_motion_data = {
                                'transl': [],
                                'fullpose': [],
                                'fullpose_rotmat': [],
                                'dataset': [],
                                'betas': [],

                                'gender': [],

                                'joints':[],
                                'verts':[],
                                'verts_full':[],
                                'verts_obj':[],
                                'velocity':[],

                                'transl_obj': [],
                                'global_orient_obj':[],
                                'global_orient_rotmat_obj': [],

                                'joints2grnd':[],
                                'joints2goal':[],
                                'verts2goal':[],
                                'joints2obj': [],
                                'verts2obj': [],
                                'rh2obj': [],

                                'bps_obj_glob':[],
                                'bps_rh_glob':[],
                                'bps_obj_rh':[],
                                'bps_rh_rh':[],
                                'contact':[],
                                'intent':[],
                                }

            for sequence in tqdm(self.split_seqs[split]):
                seq_data = parse_npz(sequence)

                obj_name = seq_data.obj_name
                sbj_id   = seq_data.sbj_id
                intent   = seq_data.motion_intent

                n_comps  = seq_data.n_comps
                gender   = seq_data.gender

                frame_mask = self.filter_contact_frames(seq_data)

                T = frame_mask.sum()
                if T < 1:
                    continue # if no frame is selected continue to the next sequence

                ##### motion data preparation

                bs = T

                grasp_motion_data['dataset'].append(np.zeros(bs, dtype=np.float32))
                grasp_motion_data['gender'].append(np.full(bs, 1 if gender == 'male' else 0, dtype=np.float32))
                sbj_vtemp = self.load_sbj_verts(sbj_id, seq_data)

                grasp_motion_data['betas'].append(np.repeat(self.sbj_info[sbj_id]['betas'], bs, axis=0))

                with torch.no_grad():
                    sbj_m = smplx.create(model_path=cfg.model_path,
                                         model_type='smplx',
                                         gender=gender,
                                         num_pca_comps=n_comps,
                                         v_template=sbj_vtemp,
                                         batch_size=bs)

                    root_offset = smplx.lbs.vertices2joints(sbj_m.J_regressor, sbj_m.v_template.view(1, -1, 3))[0, 0]

                    ##### batch motion data selection

                    sbj_params = prepare_params(seq_data.body.params, frame_mask)

                    contact_data_orig = seq_data.contact.body[frame_mask]

                    sbj_params_orig = params2torch(sbj_params)


                    ############# make relative

                    R_v2s = torch.tensor(
                        [[1., 0., 0.],
                         [0., 0., -1.],
                         [0., 1., 0.]]).reshape(1, 3, 3)
                    R = R_v2s.transpose(1, 2)

                    fpose_sbj_rotmat = aa2rotmat(sbj_params_orig['fullpose'])
                    global_orient_sbj_rel = rotmul(R, fpose_sbj_rotmat[:, 0])

                    trans_sbj_rel = rotate((sbj_params_orig['transl'] + root_offset), R) - root_offset

                    motion_sbj = {k:v.clone() for k,v in sbj_params_orig.items()}
                    global_orient_sbj_rel = self.align_z_projection_to_global_z(global_orient_sbj_rel.squeeze())

                    #####

                    motion_sbj['transl'] = to_tensor(trans_sbj_rel)
                    motion_sbj['global_orient'] = rotmat2aa(to_tensor(global_orient_sbj_rel).squeeze()).squeeze()
                    motion_sbj['global_orient_rotmat'] = to_tensor(global_orient_sbj_rel)
                    motion_sbj['fullpose'][:, :3] = motion_sbj['global_orient']
                    motion_sbj['fullpose_rotmat'] = fpose_sbj_rotmat

                    sbj_output = sbj_m(**motion_sbj)
                    verts_sbj = sbj_output.vertices
                    joints_sbj = sbj_output.joints

                    wrist_transl = joints_sbj[:, 21].clone()
                    wrist_transl[:, 1] = 0

                    motion_sbj['transl'] -= wrist_transl

                    sbj_output = sbj_m(**motion_sbj)
                    verts_sbj = sbj_output.vertices
                    joints_sbj = sbj_output.joints
                    motion_sbj['fullpose_rotmat'] = aa2rotmat(motion_sbj['fullpose'])
                    append2dict(grasp_motion_data,motion_sbj)
                    grasp_motion_data['joints'].append(to_cpu(joints_sbj))
                    grasp_motion_data['verts'].append(to_cpu(verts_sbj[:, verts_ids]))

                    frame_names.extend(['%s_%s' % (sequence.split('.')[0], fId) for fId in np.arange(T)])

            for sequence in tqdm(self.split_seqs_circle[split]):
                seq_data = parse_npz_circle(sequence)
                json_path = os.path.join(os.path.dirname(sequence), 'vr_data.json')
                with open(json_path) as json_file:
                    json_data = json.load(json_file)
                seq_data.update(json_data)

                n_comps  = 45
                gender   = seq_data.gender.item()

                n_frames = seq_data.trans.shape[0]
                goal_frame=  seq_data['goal_frame']
                frame_mask = np.array([True if not i%4 else False for i in range(n_frames)])
                frame_mask1 = np.array([True if np.abs(goal_frame-i)<120 else False for i in range(n_frames)])
                frame_mask = frame_mask1*frame_mask
                T = frame_mask.sum()
                if T < 1:
                    continue # if no frame is selected continue to the next sequence

                ##### motion data preparation

                bs = T
                grasp_motion_data['dataset'].append(np.ones(bs, dtype=np.float32))
                grasp_motion_data['gender'].append(np.full(bs, 1 if gender == 'male' else 0, dtype=np.float32))
                grasp_motion_data['betas'].append(np.repeat(seq_data.betas[:10].astype(np.float32).reshape(1,10), bs, axis=0).reshape(-1,10))

                sbj_betas = torch.tensor(seq_data.betas[:10]).unsqueeze(0).to(torch.float32)

                with torch.no_grad():
                    sbj_m = smplx.create(model_path=cfg.model_path,
                                         model_type='smplx',
                                         gender=gender,
                                         num_pca_comps=n_comps,
                                         betas = sbj_betas,
                                         batch_size=bs)

                    root_offset = smplx.lbs.vertices2joints(sbj_m.J_regressor, sbj_m.v_template.view(1, -1, 3))[0, 0]


                    ##### batch motion data selection

                    sbj_params = {
                        'global_orient': seq_data['root_orient'][frame_mask],
                        'body_pose': seq_data['pose_body'][frame_mask],
                        'transl': seq_data['trans'][frame_mask],
                        'left_hand_pose': seq_data['pose_hand'][:,:45][frame_mask],
                        'right_hand_pose': seq_data['pose_hand'][:,45:][frame_mask],
                        'jaw_pose': seq_data['pose_jaw'][frame_mask],
                        'leye_pose': seq_data['pose_eye'][:,:3][frame_mask],
                        'reye_pose': seq_data['pose_eye'][:,3:][frame_mask],
                    }

                    sbj_params_orig = params2torch(sbj_params)

                    ############# make relative

                    R_v2s = torch.tensor(
                        [[1., 0., 0.],
                         [0., 0., -1.],
                         [0., 1., 0.]]).reshape(1, 3, 3)

                    global_orient_sbj_rel = self.align_z_projection_to_global_z(aa2rotmat(sbj_params_orig['global_orient']).squeeze())


                    sbj_params_orig['fullpose'] = seq_data['poses'][frame_mask]
                    sbj_params_orig['global_orient'] = rotmat2aa(to_tensor(global_orient_sbj_rel).squeeze()).squeeze()
                    sbj_params_orig['global_orient_rotmat'] = to_tensor(global_orient_sbj_rel)
                    sbj_params_orig['fullpose'][:, :3] = sbj_params_orig['global_orient']
                    sbj_params_orig['fullpose_rotmat'] = aa2rotmat(sbj_params_orig['fullpose'])

                    sbj_output = sbj_m(**sbj_params_orig)
                    verts_sbj = sbj_output.vertices
                    joints_sbj = sbj_output.joints

                    wrist_transl = joints_sbj[:, 21].clone()
                    wrist_transl[:, 1] = 0

                    sbj_params_orig['transl'] -= wrist_transl

                    sbj_output = sbj_m(**sbj_params_orig)
                    verts_sbj = sbj_output.vertices
                    joints_sbj = sbj_output.joints


                    append2dict(grasp_motion_data,sbj_params_orig)
                    grasp_motion_data['joints'].append(to_cpu(joints_sbj))
                    grasp_motion_data['verts'].append(to_cpu(verts_sbj[:, verts_ids]))

                    frame_names.extend(['%s_%s' % (sequence.split('.')[0], fId) for fId in np.arange(T)])


            self.logger('Processing for %s split finished' % split)
            self.logger('Total number of frames for %s split is:%d' % (split, len(frame_names)))

            out_data = [grasp_motion_data]
            out_data_name = ['grasp_motion_data']

            import _pickle as pickle
            for idx, _ in enumerate(out_data):
                data_name = out_data_name[idx]
                out_data[idx] = torch2np(out_data[idx])
                outfname = makepath(os.path.join(self.save_path, split, '%s.npy' % data_name), isfile=True)

                pickle.dump(out_data[idx], open(outfname, 'wb'), protocol=4)
                out_data[idx] = 0


            np.savez(os.path.join(self.save_path, split, 'frame_names.npz'), frame_names=frame_names)

            np.save(os.path.join(self.save_path, 'obj_info.npy'), self.obj_info)
            np.save(os.path.join(self.save_path, 'sbj_info.npy'), self.sbj_info)

    def process_sequences(self):

        for sequence in self.all_seqs:
            subject_id = sequence.split('/')[-2]
            action_name = os.path.basename(sequence)
            object_name = action_name.split('_')[0]

            # filter data based on the motion intent

            if 'all' in self.intent:
                pass
            elif 'use' in self.intent and any(intnt in action_name for intnt in INTENTS[:3]):
                continue
            elif all([item not in action_name for item in self.intent]):
                continue

            # group motion sequences based on objects
            if object_name not in self.obj_based_seqs:
                self.obj_based_seqs[object_name] = [sequence]
            else:
                self.obj_based_seqs[object_name].append(sequence)

            # group motion sequences based on subjects
            if subject_id not in self.sbj_based_seqs:
                self.sbj_based_seqs[subject_id] = [sequence]
            else:
                self.sbj_based_seqs[subject_id].append(sequence)

            # split train, val, and test sequences
            self.selected_seqs.append(sequence)
            if object_name in self.splits['test']:
                self.split_seqs['test'].append(sequence)
            elif object_name in self.splits['val']:
                self.split_seqs['val'].append(sequence)
            else:
                self.split_seqs['train'].append(sequence)
                if object_name not in self.splits['train']:
                    self.splits['train'].append(object_name)


    def align_z_projection_to_global_z(self, batch_rotation_matrices):
        # Check if the input is a batch of rotation matrices
        if not (batch_rotation_matrices.dim() == 3 and
                batch_rotation_matrices.size(1) == 3 and
                batch_rotation_matrices.size(2) == 3):
            raise ValueError("Input must be a batch of 3x3 rotation matrices")

        # Prepare a list to store the adjusted matrices
        adjusted_matrices = []

        # Process each matrix in the batch
        for rot_matrix in batch_rotation_matrices:
            # Extract the local Z-axis vector from the matrix
            local_z_axis = rot_matrix[:, 2]

            # Project this vector onto the global XZ plane (set y component to 0)
            projected_z = torch.tensor([local_z_axis[0], 0, local_z_axis[2]], dtype=rot_matrix.dtype)

            # Normalize the projected vector
            projected_z /= projected_z.norm()

            # Calculate the angle between the projected vector and the global Z-axis
            angle_to_z_axis = torch.atan2(projected_z[0], projected_z[2])

            # Create a rotation matrix for this angle around the global Y-axis
            cos_a, sin_a = torch.cos(-angle_to_z_axis), torch.sin(-angle_to_z_axis)
            y_rot_matrix = torch.tensor([[cos_a, 0, sin_a],
                                         [0, 1, 0],
                                         [-sin_a, 0, cos_a]], dtype=rot_matrix.dtype)

            # Apply this rotation to the original matrix
            adjusted_matrix = y_rot_matrix @ rot_matrix
            adjusted_matrices.append(adjusted_matrix)

        # Convert the list of adjusted matrices to a tensor
        adjusted_matrices_tensor = torch.stack(adjusted_matrices)

        return adjusted_matrices_tensor
    def filter_contact_frames(self,seq_data):

        table_height = seq_data.object.params.transl[0, 2]
        table_xy = seq_data.object.params.transl[0, :2]
        obj_height = seq_data.object.params.transl[:, 2]
        obj_xy = seq_data.object.params.transl[:, :2]

        contact_array = seq_data.contact.object
        fil2 = np.logical_or((obj_height>table_height+ .005), (obj_height<table_height- .005))
        fil21 = np.logical_and((obj_height>table_height - .15), (obj_height<table_height + .15))

        fil22 = np.sqrt(np.power(obj_xy-table_xy, 2).sum(-1)) < 0.10

        include_fil = np.isin(contact_array, self.cfg.include_joints).any(axis=1)
        exclude_fil = ~np.isin(contact_array, self.cfg.exclude_joints).any(axis=1)
        fil3 = np.logical_and(include_fil, exclude_fil)
        in_contact_frames = fil2*fil21*fil22*fil3

        return in_contact_frames

    def load_sbj_verts(self, sbj_id, seq_data):

        mesh_path = os.path.join(self.grab_path, '..', seq_data.body.vtemp)
        if sbj_id in self.sbj_info:
            sbj_vtemp = self.sbj_info[sbj_id]['vtemp']
        else:
            sbj_vtemp = np.array(read_o3d_mesh(mesh_path).vertices)
            sbj_betas = np.load(mesh_path.replace('.ply', '_betas.npy'))
            self.sbj_info[sbj_id] = {'vtemp': sbj_vtemp,
                                     'gender': seq_data.gender,
                                     'betas': sbj_betas}
        return sbj_vtemp

def loc2vel(loc,fps):
    B = loc.shape[0]
    idxs = [0] + list(range(B-1))
    vel = (loc[1:] - loc[:-1])/(1/float(fps))
    return vel[idxs]

