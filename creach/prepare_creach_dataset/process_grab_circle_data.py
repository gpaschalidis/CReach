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
import sys
sys.path.append(".")
sys.path.append("..")
import os
import argparse
from tools.utils import makepath, makelogger
from tools.cfg_parser import Config
from data.creach_dataset import CReachDataset

def main():
    parser = argparse.ArgumentParser(
        description="Process GRAB and CIRCLE datasets"
    )   
    parser.add_argument(
        "--grab_path",
        required=True,
        help="The path to the GRAB dataset."
    )   
    parser.add_argument(
        "--circle_path",
        required=True,
        help="The path to the CIRCLE dataset."
    )   
    parser.add_argument(
        "--save_path",
        required=True,
        help="The type of grasp which will be used to calculate the ReachingField."
    )   
    parser.add_argument(
        "--smplx_path",
        required=True,
        help="The path to the smplx model."
    )   

    args = parser.parse_args()


    grab_path = args.grab_path
    circle_path = args.circle_path
    save_path = args.save_path
    smplx_path = args.smplx_path

    # split the dataset based on the objects
    grab_splits = {'test': ['mug', 'camera', 'binoculars', 'apple', 'toothpaste'],
                   'val': ['fryingpan', 'toothbrush', 'elephant', 'hand'],
                   'train': []}


    cfg = {

        'intent':['all'], # from 'all', 'use' , 'pass', 'lift' , 'offhand'

        'save_contact': False, # if True, will add the contact info to the saved data
        'fps':30.,
        'past':10, #number of past frames to include
        'future':10, #number of future frames to include
        ### splits
        'splits':grab_splits,

        ###IO path
        'grab_path': grab_path,
        'circle_path': circle_path,
        'save_path': save_path,

        ### number of vertices samples for each object
        'n_verts_sample': 2048,

        ### body and hand model path
        'model_path':smplx_path,
        
        ### include/exclude joints
        'include_joints' : list(range(41, 53)),
        # 'required_joints' : [16],  # mouth
        'required_joints' : list(range(53, 56)),  # thumb
        'exclude_joints' : list(range(26, 41)),
        
        ### bps info
        'r_obj' : .15,
        'n_obj': 1024,

        'r_sbj': 1.5,
        'n_sbj': 1024,
        'g_size':20,
        'h_sbj':2.,

        'r_rh': .2,
        'n_rh': 1024,

        'r_hd': .15,
        'n_hd': 2048,

        ### interpolaton params
        'interp_frames':60,

    }

    default_cfg_path = os.path.join(save_path, '../configs/grab_preprocessing_cfg.yaml')
    cfg = Config(default_cfg_path=default_cfg_path, **cfg)
    makepath(cfg.save_path)
    cfg.write_cfg(write_path=cfg.save_path+'/grab_preprocessing_cfg.yaml')

    log_dir = os.path.join(cfg.save_path, 'grab_processing.log')
    logger = makelogger(log_dir=log_dir, mode='a').info

    CReachDataset(cfg, logger)


if __name__ == '__main__':
    main()

