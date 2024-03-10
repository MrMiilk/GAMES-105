'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-03-04 10:19:37
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2024-03-10 12:42:02
FilePath: /games105/lab1/Lab2_IK_answers.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from scipy.spatial.transform import Rotation as R


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()
    joint_parent = meta_data.joint_parent
    target_pose_t = torch.tensor(target_pose)
    joint_offset = [
        meta_data.joint_initial_position[i] - 
        meta_data.joint_initial_position[joint_parent[i]]
        for i in range(len(joint_positions))]
    joint_offset_t = [torch.tensor(item) for item in joint_offset]
    joint_rot_t = [
        torch.tensor(
            (R.from_quat(joint_orientations[i]).inv().as_quat() *
            R.from_quat(joint_orientations[joint_parent[i]])).as_matrix(), 
        requires_grad=True)
        for i in range(len(joint_positions))
    ]
    joint_positions_t = [torch.tensor(item) for item in joint_positions]
    joint_orientations_t = [torch.tensor(R.from_quat(item).as_matrix()) for item in joint_orientations]
    criterion = nn.MSELoss()
    optimizer = optim.SGD(joint_rot_t, lr=0.01)
    for _ in range(500):
        for i, joint in enumerate(path):
            if i == 0: continue
            # last joint is parent of this joint
            if path[i-1] == joint_parent[joint]:
                joint_orientations_t[joint] = \
                    joint_orientations_t[joint_parent[joint]] @ joint_rot_t[joint]
                joint_positions_t[joint] = \
                    joint_positions_t[joint_parent[joint]] + \
                    joint_offset_t[joint] @ joint_orientations_t[joint_parent[joint]].T
            # this joint is parent of last joint
            else: 
                joint_orientations_t[joint] = \
                    joint_rot_t[path[i-1]].T @ joint_orientations_t[path[i-1]]
                joint_positions_t[joint] = \
                    joint_positions_t[path[i-1]] - \
                    joint_offset_t[path[i-1]] @ joint_orientations_t[joint].T
        loss = criterion(joint_positions_t[path[-1]], target_pose_t)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print("loss: ", loss)
    joint_orientations[0] = R.from_matrix(joint_rot_t[0].detach().numpy()).as_quat()
    for i in range(1, len(joint_positions)):
        joint_orientations[i] = (R.from_quat(joint_orientations[joint_parent[i]]) * \
            R.from_matrix(joint_rot_t[i].detach().numpy())).as_quat()
        joint_positions[i] = joint_positions[joint_parent[i]] + \
            joint_offset[i] @ R.from_quat(joint_orientations[joint_parent[i]]).as_matrix().T
    
    
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    
    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations