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
sys.path.append("creach")
import open3d as o3d
import torch
import numpy as np
import smplx
import os
import argparse
from creach.models.cvae import CReach
from smplx import SMPLXLayer
from reachingfield.reachingfield import ReachingField

from creach.test.reaching_body import ReachingBody
from creach.test.vis_utils import read_o3d_mesh, create_o3d_mesh, create_o3d_box_mesh

def main():
    parser = argparse.ArgumentParser(
        description="Generate Reaching bodies with CReach"
    )
    parser.add_argument(
        "--obj_rec_conf",
        required=True,
        help="The name of the receptacle configuration that is going to be used."
    ) 
    parser.add_argument(
        "--save_path",
        help="The path to the save folder"
    )
    parser.add_argument(
        "--gender",
        default="male",
        choices=["male","female","neutral"],
        help="Specify the gender"
    )  
    parser.add_argument(
        "--num_samples",
        default=1,
        type=int,
        help="Specify the batch size"
    )
    parser.add_argument(
        "--grasp_type",
        default="right",
        help="Specify the reaching body with which hand to reach the given point"
    )
    parser.add_argument(
        "--vis",
        action="store_true",
        help="Specify if you want to visualize the generated reaching bodies"
    )
    parser.add_argument(
        "--smplx_path",
        help="Provide the path to the smplx_path model"
    )
    
    args = parser.parse_args()
    obj_rec_conf = args.obj_rec_conf
    num_samples = args.num_samples
    gender = args.gender
    grasp_type = args.grasp_type
    vis = args.vis    
    smplx_path = args.smplx_path  
     
 
    save_path = args.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    network = CReach().to(device)
    network.load_state_dict(torch.load("pretrained/creach.pt", map_location=device), strict=False)
    network.eval()
   
    obj_name = obj_rec_conf.split("_")[0]
    rec_name = "_".join(obj_rec_conf.split("_")[1:-2])
    obj_path = os.path.join(os.getcwd(),"reaching_data/contact_meshes", "{}.ply".format(obj_name))
    metadata_path = os.path.join(os.getcwd(),"reaching_data/replicagrasp", "dset_info.npz")
    metadata = dict(np.load(metadata_path, allow_pickle=True))[obj_rec_conf]
    obj_transl = metadata[0]
    global_orient = metadata[1]
    obj_mesh = read_o3d_mesh(obj_path)
    obj_verts = np.array(obj_mesh.vertices) @ global_orient.T + obj_transl
    obj_center = (obj_verts.max(0) + obj_verts.min(0)) / 2
    obj_faces = np.array(obj_mesh.triangles)
    obj_mesh = create_o3d_mesh(obj_verts, obj_faces, (0.3, 0.8, 0.1))

    rec_data_path = os.path.join(os.getcwd(),"reaching_data/replicagrasp", "receptacles.npz")
    rec_data = dict(np.load(rec_data_path, allow_pickle=True))[rec_name]
    rec_verts = rec_data[0][0]
    rec_faces = rec_data[0][1]

    rec_mesh = create_o3d_mesh(rec_verts, rec_faces, (0.3,0.1,0.5))
    floor_mesh = create_o3d_box_mesh(rec_verts)
    reachingfield = ReachingField(obj_path)
    ray_dirs, hor_ray_dirs = reachingfield.sample(
                obj_transl,
                global_orient,
                [rec_mesh],
                grasp_type=grasp_type,
                num_samples=num_samples
    )

    
    reaching_body = ReachingBody(
        network, obj_mesh, obj_center, metadata, [rec_mesh, floor_mesh], smplx_path, device
    )
    body_params = reaching_body.generate(
        num_samples, grasp_type, ray_dirs, hor_ray_dirs, gender, vis
    ) 
   
    results = {} 
    for key,val in body_params.items():
        results[key] = val.cpu().numpy()
    
    np.savez(os.path.join(save_path, "results.npz"),
             body_params=results
    )


if __name__ == "__main__":
    main()


