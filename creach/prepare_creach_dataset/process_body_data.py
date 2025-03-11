import sys
sys.path.append("..")
import argparse
import open3d as o3d
import numpy as np
import smplx
from smplx import SMPLXLayer
import numpy as np
import torch
import os
from tqdm import tqdm
import pickle as pkl
import shutil
import random

from creach.test.vis_utils import create_o3d_mesh, rotmat2aa, aa2rotmat


def temp(fullpose_rotmat, M, list_indexes):
    for left_index, right_index in list_indexes:
        temp = fullpose_rotmat[:, left_index, :, :].copy()
        fullpose_rotmat[:, left_index, :, :] = fullpose_rotmat[:, right_index, :, :]
        fullpose_rotmat[:, right_index, :, :] = temp
    return fullpose_rotmat


def main():
    parser = argparse.ArgumentParser(
        description="""Find the left arm reaching bodies inside the CIRLE data, mirror them, and 
            move the bodies so they touch the ground. The new dataset will contain only 
            right arm reaching bodies."""
    )
    parser.add_argument(
        "--dataset_path",
        required=True,
        help="The path of the dataset"
    )
    parser.add_argument(
        "--smplx_path",
        required=True,
        help="The path of the smplx model"
     )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="The path of the mirrored dataset"
    )
    parser.add_argument(
        "--sampled_verts_ids",
        required=True,
        help="The path to the .npy file that contains the ids of the sampled vertices"
    )   
    args = parser.parse_args()


    data_path = args.dataset_path
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    sampled_mask = np.load(args.sampled_verts_ids)

    splits = ["test", "val", "train"]    
    
    for sp in splits:
        print("Start processing {} split".format(sp))
        data = dict(np.load(
            os.path.join(args.dataset_path, sp, "grasp_motion_data.npy"), 
            allow_pickle=True
            )
        )

        ##################################
        #Find the left arm reaching bodies
        ##################################
        head_front, head_back = data["verts"][:,386,:], data["verts"][:,387,:]
        head_vec = head_front-head_back
        gaze_dir = head_vec / np.sqrt((head_vec**2).sum(1))[...,None]

        forehead = (data["joints"][:,23] + data["joints"][:,24])/2 # The point between the eyes
        
        right_wrists = data["joints"][:,21] - forehead
        rh_wrist = right_wrists / np.sqrt((right_wrists**2).sum(1))[...,None]


        left_wrists = data["joints"][:,20] - forehead
        lh_wrist = left_wrists / np.sqrt((left_wrists**2).sum(1))[...,None]

        rh_queries = (gaze_dir*rh_wrist).sum(1)
        lh_queries = (gaze_dir*lh_wrist).sum(1)
        
        rh_angs=np.arccos(rh_queries)
        lh_angs=np.arccos(lh_queries)

        ########################################################
        # We are looking for left hand grasps in the CIRCLE data
        ########################################################

        left_grasp_mask = (lh_angs<rh_angs)*(data["dataset"]==1)
        ###########################################################
        #Instead of deleting these data we are going to mirror them
        ###########################################################
        
        left_fullpose_rotmat = data["fullpose_rotmat"][left_grasp_mask].copy()
        left_transl = data["transl"][left_grasp_mask].copy()
        M = np.eye(3)
        M[0][0] = -1
        M = M.astype(np.float32)
        right_fullpose_rotmat = M @ left_fullpose_rotmat @ M
        right_fullpose_rotmat = temp(right_fullpose_rotmat, M, [[1,2],[4,5],[7,8],[10,11],[13,14],[16,17],[18,19],[20,21], [23,24],[range(25,40), range(40,55)]])
        
        left_transl[:,0] *= -1
        right_transl = left_transl
        right_fullpose = rotmat2aa(torch.tensor(right_fullpose_rotmat)).squeeze().reshape(-1,165).numpy()

        fullpose = data["fullpose"].copy()
        fullpose[left_grasp_mask] = right_fullpose
        fullpose_rotmat = data["fullpose_rotmat"].copy()
        fullpose_rotmat[left_grasp_mask] = right_fullpose_rotmat
        transl = data["transl"].copy()            
        transl[left_grasp_mask] = right_transl
     

        female_model = SMPLXLayer(
            model_path=os.path.join(args.smplx_path,"smplx","SMPLX_FEMALE.npz"),
            gender='female',
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(device)

        male_model = SMPLXLayer(
            model_path=os.path.join(args.smplx_path,"smplx","SMPLX_MALE.npz"),
            gender='male',
            num_pca_comps=45,
            flat_hand_mean=True,
        ).to(device)

        sampled_verts = []
        total_transl = []
        total_joints = []
        
        for i in tqdm(range(len(data["fullpose"]))):
            body_dict = {"body_pose":torch.tensor(fullpose_rotmat[i, 1:22, :, :]).unsqueeze(0).to(device),
                         "global_orient":torch.tensor(fullpose_rotmat[i, 0, :, :]).unsqueeze(0).to(device),
                         "jaw_pose":torch.tensor(fullpose_rotmat[i, 22, :, :]).unsqueeze(0).to(device),
                         "leye_pose":torch.tensor(fullpose_rotmat[i, 23, :, :]).unsqueeze(0).to(device),
                         "reye_pose":torch.tensor(fullpose_rotmat[i, 24, :, :]).unsqueeze(0).to(device),
                         "left_hand_pose":torch.tensor(fullpose_rotmat[i, 25:40, :, :]).unsqueeze(0).to(device),
                         "right_hand_pose":torch.tensor(fullpose_rotmat[i, 40:55, :, :]).unsqueeze(0).to(device),
                         "transl": torch.tensor(transl[i]).unsqueeze(0).to(device),
                         "betas": torch.tensor(data["betas"][i]).unsqueeze(0).to(device)
            }

            if data["gender"][i] == 0:
                body = female_model(**body_dict)
            else:
                body = male_model(**body_dict)
                
            verts = body.vertices.detach().squeeze().cpu().numpy()
            sorted_indices = np.argsort(verts[:,1])
            verts =  verts[sorted_indices]
            trans = transl[i] - verts[0]
           

            body_dict = {"body_pose":torch.tensor(fullpose_rotmat[i, 1:22, :, :]).unsqueeze(0).to(device),
                         "global_orient":torch.tensor(fullpose_rotmat[i, 0, :, :]).unsqueeze(0).to(device),
                         "jaw_pose":torch.tensor(fullpose_rotmat[i, 22, :, :]).unsqueeze(0).to(device),
                         "leye_pose":torch.tensor(fullpose_rotmat[i, 23, :, :]).unsqueeze(0).to(device),
                         "reye_pose":torch.tensor(fullpose_rotmat[i, 24, :, :]).unsqueeze(0).to(device),
                         "left_hand_pose":torch.tensor(fullpose_rotmat[i, 25:40, :, :]).unsqueeze(0).to(device),
                         "right_hand_pose":torch.tensor(fullpose_rotmat[i, 40:55, :, :]).unsqueeze(0).to(device),
                         "transl": torch.tensor(trans).unsqueeze(0).to(device),
                         "betas": torch.tensor(data["betas"][i]).unsqueeze(0).to(device)
            }
            
            
            if data["gender"][i] == 0:
                body = female_model(**body_dict)
            else:
                body = male_model(**body_dict)
            verts = body.vertices.detach().squeeze().cpu().numpy()
            sampled_verts.append(verts[sampled_mask])
            total_transl.append(trans)
            joints = body.joints.squeeze().cpu().numpy()[:55,:]
            total_joints.append(joints)

        sampled_verts = np.array(sampled_verts)
        total_transl = np.array(total_transl)
        total_joints = np.array(total_joints)
        final_dict ={
            "transl":total_transl,
            "fullpose":fullpose,
            "fullpose_rotmat":fullpose_rotmat,
            "dataset":data["dataset"],
            "betas":data["betas"],
            "gender":data["gender"],
            "joints":total_joints,
            "verts":sampled_verts
        }

    
        save_dir = os.path.join(args.save_dir, sp) 
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
         
        with open(os.path.join(save_dir, "grasp_motion_data.npy"), "wb") as f:
            pkl.dump(final_dict, f)


        shutil.copy(
            os.path.join(args.dataset_path, sp, "frame_names.npz"), 
            os.path.join(save_dir, "frame_names.npz")
        )
        print("Processing of {} split is complete".format(sp))
    
if __name__ == "__main__":
    main()
