import numpy as np
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data


def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_names: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()

        joint_name = []
        joint_parent = []
        joint_offset = []
        stack = []  # using stack to track parent index for each joint
        # it is a tree seq question
        for i in range(len(lines)):
            if lines[i].startswith('ROOT'):
                joint_name.append(lines[i].split()[1])
                joint_parent.append(-1)
            elif lines[i].startswith('MOTION'):
                break
            else:
                tmp_line = lines[i].split()
                if tmp_line[0] == '{':
                    stack.append(len(joint_name)-1)  # parent index is joint_name[-1]
                elif tmp_line[0] == '}':
                    stack.pop()
                elif tmp_line[0] == 'JOINT':
                    joint_name.append(tmp_line[1])
                    joint_parent.append(stack[-1])
                elif tmp_line[0] == 'End':  # align push and pop operation
                    joint_name.append(joint_name[stack[-1]]+'_end')
                    joint_parent.append(stack[-1])
                elif tmp_line[0] == 'OFFSET':
                    joint_offset.append(np.array([float(x) for x in tmp_line[1:4]]).reshape(1, -1))
                else:
                    continue

    joint_offset = np.concatenate(joint_offset, axis=0)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """
    motion = motion_data[frame_id]
    joint_positions = [motion[:3]]
    joint_orientations = [R.from_euler('XYZ', motion[3:6], degrees=True)]
    cnt = 2
    for i, joint in enumerate(joint_name):
        if i == 0: continue
        if joint.endswith('_end'): rot = np.array([0, 0, 0])
        else: 
            rot = motion[3*cnt: 3*cnt+3]
            cnt += 1
        ori = R.from_euler('XYZ', rot, degrees=True)
        ori_par = joint_orientations[joint_parent[i]]
        trans_par = joint_positions[joint_parent[i]]
        ori_now = ori_par * ori
        trans_now = ori_par.as_matrix() @ joint_offset[i] + trans_par
        joint_positions.append(trans_now)
        joint_orientations.append(ori_now)
    joint_orientations = [ori.as_quat() for ori in joint_orientations]
    return np.array(joint_positions), np.array(joint_orientations)


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    motion_data = []
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)
    A_motion_data = load_motion_data(A_pose_bvh_path)
    
    A_idx_map = {}
    padding = 0
    for i, joint in enumerate(A_joint_name):
        if joint.endswith('_end'):
            padding += 1
        A_idx_map[joint] = i - padding
    
    lShoulder_TA = R.from_euler('XYZ', [0, 0, -45], degrees=True)
    rShoulder_TA = R.from_euler('XYZ', [0, 0, 45], degrees=True)
    for motion in A_motion_data:
        new_motion = []
        for T_joint in T_joint_name:
            A_idx = A_idx_map[T_joint]
            if T_joint == 'RootJoint':
                assert A_idx == 0
                new_motion.append(motion[0:3])
            if T_joint == 'lShoulder':
                ori_A = R.from_euler('XYZ', motion[A_idx*3+3: A_idx*3+6], degrees=True)
                ori_T = ori_A * lShoulder_TA
                new_motion.append(ori_T.as_euler('XYZ', True))
            elif T_joint == 'rShoulder':
                ori_A = R.from_euler('XYZ', motion[A_idx*3+3: A_idx*3+6], degrees=True)
                ori_T = ori_A * rShoulder_TA
                new_motion.append(ori_T.as_euler('XYZ', True))
            elif T_joint.endswith('_end'): 
                continue
            else:
                new_motion.append(motion[A_idx*3+3: A_idx*3+6])
            
        motion_data.append(np.concatenate(new_motion))
    
    return np.stack(motion_data)
