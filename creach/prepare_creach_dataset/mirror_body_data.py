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

from creach.test.vis_utils import rotmat2aa, aa2rotmat


def temp(fullpose_rotmat, M, list_indexes):
    for left_index, right_index in list_indexes:
        temp = fullpose_rotmat[:, left_index, :, :].copy()
        fullpose_rotmat[:, left_index, :, :] = fullpose_rotmat[:, right_index, :, :]
        fullpose_rotmat[:, right_index, :, :] = temp
    return fullpose_rotmat


def main():
    parser = argparse.ArgumentParser(
        description="Mirror body data and create a dataset with both hand and left arm reaching bodies"
    )

    parser.add_argument(
        "--dataset_path",
        required=True,
        help="The path of the dataset"
    )
    parser.add_argument(
        "--smplx_path",
        required=True,
        help="The path of the dataset"
    )
    parser.add_argument(
        "--save_dir",
        required=True,
        help="The path of the mirrored dataset"
    )
    parser.add_argument(
        "--sampled_verts_ids",
        default="consts/verts_ids_0512.npy",
        help="The path to the .npy file that contains the ids of the sampled vertices"
    )
    args = parser.parse_args()


    data_path = args.dataset_path
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
      
    splits = ["test", "val", "train"]

    for sp in splits:
        print("Start processing {} split".format(sp))
        sampled_mask = np.load(args.sampled_verts_ids)

        data = dict(np.load(os.path.join(args.dataset_path, sp, "grasp_motion_data.npy"), allow_pickle=True))
        
        fullpose_rotmat = data["fullpose_rotmat"].copy()

        M = np.eye(3)
        M[0][0] = -1
        M = M.astype(np.float32)
        fullpose_rotmat = M @ fullpose_rotmat @ M
        fullpose_rotmat = temp(fullpose_rotmat, M, [[1,2],[4,5],[7,8],[10,11],[13,14],[16,17],[18,19],[20,21], [23,24],[range(25,40), range(40,55)]])
        mirr_transl = data["transl"].copy()
        mirr_transl[:,0] *= -1
        fullpose = rotmat2aa(torch.tensor(fullpose_rotmat)).squeeze().reshape(-1,165).numpy()
        
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

        total_joints = []
        sampled_verts = []   
        for i in tqdm(range(len(data["fullpose"]))):

            body_dict = {"body_pose":torch.tensor(fullpose_rotmat[i, 1:22, :, :]).unsqueeze(0).to(device),
                         "global_orient":torch.tensor(fullpose_rotmat[i, 0, :, :]).unsqueeze(0).to(device),
                         "jaw_pose":torch.tensor(fullpose_rotmat[i, 22, :, :]).unsqueeze(0).to(device),
                         "leye_pose":torch.tensor(fullpose_rotmat[i, 23, :, :]).unsqueeze(0).to(device),
                         "reye_pose":torch.tensor(fullpose_rotmat[i, 24, :, :]).unsqueeze(0).to(device),
                         "left_hand_pose":torch.tensor(fullpose_rotmat[i, 25:40, :, :]).unsqueeze(0).to(device),
                         "right_hand_pose":torch.tensor(fullpose_rotmat[i, 40:55, :, :]).unsqueeze(0).to(device),
                         "transl": torch.tensor(mirr_transl[i]).unsqueeze(0).to(device),
                         "betas": torch.tensor(data["betas"][i]).unsqueeze(0).to(device)
            }
           
            if data["gender"][i] == 0:
                body = female_model(**body_dict)
            else:
                body = male_model(**body_dict)

            
            verts = body.vertices.detach().squeeze().cpu().numpy()
            sampled_verts.append(verts[sampled_mask])
            joints = body.joints.squeeze().cpu().numpy()[:55,:]
            total_joints.append(joints)

        total_joints = np.array(total_joints)
        sampled_verts = np.array(sampled_verts)    


        final_transl = np.vstack((data["transl"],mirr_transl))
        final_fullpose = np.vstack((data["fullpose"],fullpose))
        final_fullpose_rotmat = np.vstack((data["fullpose_rotmat"],fullpose_rotmat))
        final_dataset = np.concatenate((data["dataset"],data["dataset"]))
        final_betas = np.vstack((data["betas"],data["betas"]))
        final_gender = np.concatenate((data["gender"],data["gender"]))
        final_joints = np.vstack((data["joints"],total_joints))
        final_verts = np.vstack((data["verts"],sampled_verts))
        grasp_type = np.concatenate((np.zeros(len(data["dataset"])),np.ones(len(data["dataset"]))))


        final_dict ={
            "transl":final_transl,
            "fullpose":final_fullpose,
            "fullpose_rotmat":final_fullpose_rotmat,
            "dataset":final_dataset,
            "betas":final_betas,
            "gender":final_gender,
            "joints":final_joints,
            "verts":final_verts,
            "grasp_type":grasp_type
        }

        save_dir = os.path.join(args.save_dir, sp)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        with open(os.path.join(save_dir, "grasp_motion_data.npy"), "wb") as f:
            pkl.dump(final_dict, f)

        frames = dict(np.load(os.path.join(args.dataset_path, sp, "frame_names.npz")))
        
        frame_names = frames["frame_names"]
        mirrored_frame_names = frame_names.copy()
        mirrored_frame_names = np.array(
            ["/mirrored" + line for line in mirrored_frame_names]
        )
        
        total_frame_names = np.concatenate((frame_names,mirrored_frame_names))
        total_frames = {}
        total_frames["frame_names"] = total_frame_names 
        
        np.savez(os.path.join(save_dir, "frame_names.npz"),**total_frames)
        print("Processing of {} split is complete".format(sp))

if __name__ == "__main__":
    main()
