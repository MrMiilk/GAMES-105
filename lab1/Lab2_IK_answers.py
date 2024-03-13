'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2024-03-04 10:19:37
LastEditors: Miilk 1024109095@qq.com
LastEditTime: 2024-03-13 22:42:49
FilePath: /games105/lab1/Lab2_IK_answers.py
Description: 

Copyright (c) 2024 by ${git_name_email}, All Rights Reserved. 
'''
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from MetaData import MetaData

def orthogonal_regularization(matrix):
    """
    对矩阵进行正交正则化。
    """
    with torch.no_grad():
        u, _, v = torch.svd(matrix, some=True)
        matrix.copy_(torch.matmul(u, v.t()))
        
def quaternion_to_rotation_matrix(quaternion):
    """
    Convert a quaternion to a rotation matrix.
    Args:
        quaternion (Tensor): a tensor of quaternions of shape (..., 4).
    Returns:
        Tensor: a rotation matrix of shape (..., 3, 3).
    """
    # 正规化四元数
    quaternion = F.normalize(quaternion, p=2, dim=-1)
    
    # 提取四元数的组成部分
    x, y, z, w = quaternion[..., 0], quaternion[..., 1], quaternion[..., 2], quaternion[..., 3]
    
    # 计算旋转矩阵
    xx, yy, zz = x * x, y * y, z * z
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotation_matrix = torch.stack([
        1 - 2 * (yy + zz),     2 * (xy - wz),     2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz),     2 * (yz - wx),
            2 * (xz - wy),     2 * (yz + wx), 1 - 2 * (xx + yy)
    ], dim=-1).reshape(quaternion.shape[:-1] + (3, 3))
    
    return rotation_matrix

def rotation_matrix_to_quaternion(R):
    """
    Convert a rotation matrix to a quaternion.
    
    Parameters:
    - R: A rotation matrix of shape (3, 3).
    
    Returns:
    - A quaternion [x, y, z, w] corresponding to the rotation matrix.
    """
    if not R.shape == (3, 3):
        raise ValueError("R must be a 3x3 matrix.")
    
    # Allocate space for the quaternion
    q = torch.empty(4)
    
    # Compute the quaternion components
    q[0] = (R[2, 1] - R[1, 2]) / (4 * q[0])
    q[1] = (R[0, 2] - R[2, 0]) / (4 * q[0])
    q[2] = (R[1, 0] - R[0, 1]) / (4 * q[0])
    q[3] = 0.5 * torch.sqrt(1 + R[0, 0] + R[1, 1] + R[2, 2])
    
    return q

def rotate_point_with_quaternion(point, quaternion):
    """
    Rotate a point using a quaternion.
    """
    # Create a pure quaternion from the point
    point_quat = torch.tensor([*point, 0])
    
    # Compute the conjugate of the quaternion
    quaternion_conjugate = quaternion * torch.tensor([-1, -1, -1, 1])
    
    # Rotate the point
    point_rotated = quaternion_multiply(quaternion_multiply(quaternion, point_quat), quaternion_conjugate)
    
    # Extract the vector part
    return point_rotated[:-1]

def quaternion_multiply(q1, q2):
    """
    Multiply two quaternions to get the quaternion representing the combined rotation.
    
    Parameters:
    - q1: A quaternion represented as a tensor [x, y, z, w].
    - q2: Another quaternion represented as a tensor [x, y, z, w].
    
    Returns:
    - The resulting quaternion of the multiplication.
    """
    # Extract scalar (real) and vector (imaginary) parts
    v1, w1 = q1[:-1], q1[-1]
    v2, w2 = q2[:-1], q2[-1]
    
    # Calculate the scalar (real) part of the resulting quaternion
    w = w1 * w2 - torch.dot(v1, v2)
    
    # Calculate the vector (imaginary) part of the resulting quaternion
    v = w1 * v2 + w2 * v1 + torch.cross(v1, v2)
    
    # Combine the scalar and vector parts
    q3 = torch.cat((v, w.unsqueeze(0)))
    
    return q3

def quaternion_inverse(q):
    """
    Compute the inverse of a quaternion.
    
    Parameters:
    - q: A quaternion represented as a tensor [x, y, z, w].
    
    Returns:
    - The inverse of the quaternion.
    """
    # Assume q is a unit quaternion, its inverse is its conjugate
    # For a non-unit quaternion, you would divide by the norm squared
    q_conjugate = q * torch.tensor([-1, -1, -1, 1])
    return q_conjugate

def quaternion_loss(q):
    """额外的损失项，确保四元数接近单位四元数"""
    return (q.norm(p=2, dim=0)**2 - 1)**2

def do_optm(meta_data, joint_positions, joint_orientations, target_pose, lr=0.01):
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
            quaternion_multiply(
                quaternion_inverse(torch.tensor(joint_orientations[joint_parent[i]])), 
                torch.tensor(joint_orientations[i])
            )).clone().detach().requires_grad_(True)
        if i in path else
        torch.tensor(
            quaternion_multiply(
                quaternion_inverse(torch.tensor(joint_orientations[joint_parent[i]])), 
                torch.tensor(joint_orientations[i])
            ), requires_grad=False)
        for i in range(len(joint_positions))
    ]
    joint_positions_t = [torch.tensor(item) for item in joint_positions]
    joint_orientations_t = [torch.tensor(item) for item in joint_orientations]
    joint_orientations_t[path[0]] = \
        joint_orientations_t[path[0]].clone().detach().requires_grad_(True)
    criterion = nn.MSELoss()
    optimizer = optim.SGD(joint_rot_t, lr=lr)
    loss = torch.tensor(100)
    max_iter = 300
    while loss.item() > 1e-3 and max_iter > 0:
        max_iter = max_iter - 1
        for i, joint in enumerate(path):
            if i == 0:
                continue
            # last joint is parent of this joint
            if path[i-1] == joint_parent[joint]:
                joint_orientations_t[joint] = \
                    quaternion_multiply(
                        joint_orientations_t[joint_parent[joint]], joint_rot_t[joint]
                    )
                joint_positions_t[joint] = \
                    joint_positions_t[joint_parent[joint]] + \
                    rotate_point_with_quaternion(
                        joint_offset_t[joint], 
                        joint_orientations_t[joint_parent[joint]]
                    )
            # this joint is parent of last joint
            else: 
                joint_orientations_t[joint] = \
                    quaternion_multiply(
                        quaternion_inverse(joint_rot_t[path[i-1]]), 
                        joint_orientations_t[path[i-1]]
                    )
                joint_positions_t[joint] = \
                    joint_positions_t[path[i-1]] - \
                    rotate_point_with_quaternion(
                        joint_offset_t[path[i-1]], 
                        joint_orientations_t[joint]
                    )
        loss = criterion(joint_positions_t[path[-1]], target_pose_t)
        for item in joint_rot_t: 
            loss += 2*quaternion_loss(item)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # for item in joint_rot_t: item = F.normalize(item, p=2, dim=0)
    print("loss: ", loss)
    
    joint_orientations[0] = joint_orientations_t[0].detach().numpy()
    joint_positions[0] = joint_positions_t[0].detach().numpy()
    for i in range(1, len(joint_positions)):
        joint_orientations[i] = quaternion_multiply(
            torch.tensor(joint_orientations[joint_parent[i]]), 
            joint_rot_t[i]
        ).detach().numpy()
        joint_positions[i] = joint_positions[joint_parent[i]] + \
            rotate_point_with_quaternion(
                joint_offset_t[i], 
                torch.tensor(joint_orientations[joint_parent[i]])
            ).detach().numpy()
    
    return joint_positions, joint_orientations

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
    return do_optm(meta_data, joint_positions, joint_orientations, target_pose, lr=0.01)

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """
    tgt_posi = [joint_positions[0][0]+relative_x, target_height, joint_positions[0][2]+relative_z]
    return do_optm(meta_data, joint_positions, joint_orientations, tgt_posi, lr=0.1)
    
def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    joint_positions, joint_orientations = do_optm(meta_data, joint_positions, joint_orientations, left_target_pose, lr=0.01)
    meta_data2 = meta_data
    meta_data2.root_joint = 'rTorso_Clavicle' # 'RootJoint'
    meta_data2.end_joint = 'rWrist_end'
    joint_positions, joint_orientations = do_optm(meta_data2, joint_positions, joint_orientations, right_target_pose, lr=0.05)
    return joint_positions, joint_orientations