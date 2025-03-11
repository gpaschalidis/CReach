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
import numpy as np
import open3d as o3d
import torch
import os
import smplx
from smplx import SMPLXLayer
from .vis_utils import rotmat2aa, create_o3d_mesh, create_o3d_box_mesh,\
        create_line_set,reshaping, find_new_transl_and_gorient

from creach.models.model_utils import parms_6D2full

class ReachingBody:
    def __init__(
        self, network, obj_mesh, obj_center, metadata, around_meshes, smplx_path, device

    ):
        self.network = network
        self.obj_mesh = obj_mesh
        self.obj_center = obj_center
        self.around_meshes = around_meshes
        self.obj_transl = metadata[0]
        self.global_orient = metadata[1]
        self.smplx_path = smplx_path
        self.device = device

    def generate(self, num_samples, grasp_type, ray_dirs, hor_ray_dirs, gender, vis):
        angs = np.arccos(np.dot(hor_ray_dirs, np.array([0,1,0])))
        cross = np.cross(hor_ray_dirs, np.array([0, 1, 0]))
        ang_mask = np.dot(cross,np.array([0, 0, 1])) < 0    
    
        Rz = np.eye(3)[None]
        Rz = Rz.repeat(len(angs), 0)
        Rz[:,0,0] = np.cos(angs)
        Rz[:,1,1] = np.cos(angs)
        Rz[:,0,1] = - np.sin(angs)
        Rz[:,1,0] = np.sin(angs)
        Rz[ang_mask] = Rz[ang_mask].transpose(0,2,1)

        directions = ray_dirs.reshape(num_samples, 1, 3)
        directions = (directions @ Rz.transpose(0,2,1)).squeeze()
        directions = directions.reshape(num_samples, 3)


        proj_directions = hor_ray_dirs.reshape(num_samples, 1, 3)
        proj_directions = (hor_ray_dirs @ Rz.transpose(0,2,1)).squeeze()

        dec_x = {}

        Rx = torch.eye(3)
        Rx[1][1] = 0
        Rx[2][2] = 0
        Rx[1][2] = -1
        Rx[2][1] = 1

        obj_loc = (Rx.T @ torch.tensor(self.obj_transl.T).to(torch.float32)).T
        wrist_loc = obj_loc.clone()

        dec_x["obj_loc"] = torch.repeat_interleave(obj_loc[:,1][None], num_samples,0).to(self.device)

        directions = (torch.tensor(directions).to(torch.float32) @ Rx)
        dec_x["dir"] = directions.to(self.device)

        betas = torch.zeros(num_samples,10).to(self.device)

        dec_x["betas"] = betas

        z_enc = torch.distributions.normal.Normal(
            loc=torch.zeros([num_samples, self.network.latentD], requires_grad=False).to(self.device),
            scale=torch.ones([num_samples, self.network.latentD], requires_grad=False).to(self.device)
        )

        z_enc_s = z_enc.rsample()
        dec_x["z"] = z_enc_s


        ###### For right hand
        if grasp_type == "right":
            dec_x["grasp_type"] = torch.zeros((num_samples)).to(torch.float32).to(self.device)
        else:
        ###### For left hand
            dec_x["grasp_type"] = torch.ones((num_samples)).to(torch.float32).to(self.device)

        dec_x = torch.cat([v.reshape(num_samples, -1) for v in dec_x.values()], dim=1)

        dec_out = self.network.decode(dec_x)
        pose, trans = dec_out['pose'].detach(), dec_out['trans'].detach() + wrist_loc.to(self.device)
        d62rot = pose.shape[-1] == 330

        bparams = parms_6D2full(pose, trans, d62rot=d62rot)
        bparams["betas"] = betas

        Rx = Rx.to(self.device)

        body_model = smplx.create(
            model_path=self.smplx_path,
            model_type="smplx",
            gender=gender,
            use_pca=False,
            num_pca_comps=45,
            flat_hand_mean=True,
            batch_size=num_samples
        ).to(self.device).eval()

        faces = body_model.faces

        bm = body_model(**{"body_pose":rotmat2aa(bparams["body_pose"].cpu()).to(self.device).reshape(1,-1)})
        joints = bm.joints.detach().cpu().numpy()
        pelvis = joints[:,0,:]
        bparams["global_orient"] = (Rx @ bparams["global_orient"].squeeze()).reshape(num_samples,1,3,3)

        bparams["transl"] = torch.tensor((pelvis + bparams["transl"].cpu().numpy()) @ Rx.cpu().numpy().T - pelvis)
        rparams = reshaping(bparams, self.device)

        new_gor, new_transl = find_new_transl_and_gorient(
            Rz.transpose(0,2,1),
            bparams,
            self.obj_transl,
            pelvis,
            num_samples,
            self.device
        )

        rparams["global_orient"] = new_gor
        rparams["transl"] = new_transl
        bm = body_model(**rparams)
        final_verts = bm.vertices.detach().cpu().numpy()
        final_body_meshes = [create_o3d_mesh(final_verts[i],faces,[0.9,0.2,0.5]) for i in range(num_samples)]

        if vis:
            for i in range(num_samples):
                lset = create_line_set(self.obj_center[None], self.obj_center[None] + 2 * ray_dirs[i], (1,0,0))
                o3d.visualization.draw_geometries([lset, self.obj_mesh, final_body_meshes[i]] + self.around_meshes)
        return bparams













