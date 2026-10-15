"""
1) Copyright (C) 2024  ETH Zurich, Hsuan-I Ho
2) KWU CVLAB, SY Noh
"""
import json
import torch
from smplx import SMPL

## JOINT_MAP = 'data/smplx_openpose25.json'
SMPL_PATH = 'data/body_models/smpl'


def load_json(json_path, dataset_name='thuman', device='cuda'):
    smpl_data = json.load(open(json_path))

    param_betas = torch.tensor(smpl_data['betas'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()
    param_poses = torch.tensor(smpl_data['body_pose'], dtype=torch.float32, device=device).unsqueeze(0).contiguous()

    if dataset_name=='customhumans':
        gender = 'male'
    elif dataset_name=='thuman':
        gender = 'male'
    else:
        gender = smpl_data.get('gender', 'neutral')

    body_model = SMPL(model_path=SMPL_PATH, gender=gender).to(device)

    J_0 = body_model(body_pose = param_poses, betas=param_betas).joints.contiguous().detach()
    output = body_model(betas=param_betas,
                                   body_pose=param_poses,
                                   transl=-J_0[:,0,:],
                                   )

    V = output.vertices.detach().to(device)
    F = torch.tensor(body_model.faces.astype(int)).to(device)
    joint_3d = None

    return V, F, joint_3d
