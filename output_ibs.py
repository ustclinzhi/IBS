# 创建新文件: e:\IBS\dexmachina\output_ibs_iterations.py
import os
import torch
import numpy as np
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
import sys
sys.path.append(os.path.join(os.path.dirname(__file__)))

from dexmachina.envs.viewibs import DifferentiableIBS

def save_points_as_obj(points_tensor, filename):
    """
    保存点云张量(1, N, 3)到OBJ文件。
    """
    points_np = points_tensor.squeeze(0).cpu().numpy()
    with open(filename, 'w') as f:
        for p in points_np:
            f.write(f"v {p[0]} {p[1]} {p[2]}\n")
    print(f"已保存 {filename}")

def process_hand_object_ibs(target_directory):
    """
    处理指定目录下的右手和物体数据，输出不同迭代次数的IBS采样点云。
    """
    print(f"正在处理目录: {target_directory}")
    
    # 设置工作目录到目标目录，这样保存的文件会在这个目录下
    original_cwd = os.getcwd()
    os.chdir(target_directory)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载手部和物体文件
    right_hand_path = os.path.join(target_directory, "right_hand.obj")
    object_path = os.path.join(target_directory, "object.obj")
    
    if not os.path.exists(right_hand_path):
        print(f"错误: 找不到文件 {right_hand_path}")
        return
    if not os.path.exists(object_path):
        print(f"错误: 找不到文件 {object_path}")
        return
    
    # 加载OBJ文件
    print("正在加载OBJ文件...")
    try:
        hand_verts, hand_faces, _ = load_obj(right_hand_path)
        obj_verts, obj_faces, _ = load_obj(object_path)
        print(f"右手顶点形状: {hand_verts.shape}")
        print(f"物体顶点形状: {obj_verts.shape}")
    except Exception as e:
        print(f"加载OBJ文件时出错: {e}")
        return
    
    # 移动到设备
    hand_verts = hand_verts.to(device)
    obj_verts = obj_verts.to(device)
    
    # 初始化IBS计算器，设置最大迭代次数为足够大的值
    ibs_calculator = DifferentiableIBS(max_iter=40, n_samples=512)
    ibs_calculator = ibs_calculator.to(device)
    
    # 需要保存的迭代次数
    save_iterations = [0, 1,2, 5, 7]  # 0表示初始点
    
    # 扩展维度以匹配期望的输入形状 [B, N, 3]
    hand_batch = hand_verts.unsqueeze(0)  # [1, N_hand, 3]
    obj_batch = obj_verts.unsqueeze(0)    # [1, N_obj, 3]
    
    print("开始计算不同迭代次数的IBS点云...")
    
    # 1. 生成初始采样点
    init_points = ibs_calculator._generate_initial_points(hand_batch, obj_batch)
    if 0 in save_iterations:
        ibs_calculator.save_points_as_obj(init_points, "initial_sampling_points.obj")
    
    # 2. 进行迭代优化并保存特定迭代的结果
    ibs_points = ibs_calculator.iterative_optimization_with_save(init_points, obj_batch, hand_batch, save_iterations)
    
    # 保存最终的IBS结果
    final_ibs_path = os.path.join(target_directory, "final_ibs_points.obj")
    save_points_as_obj(ibs_points, final_ibs_path)
    
    # 恢复原始工作目录
    os.chdir(original_cwd)
    
    print("处理完成！所有迭代的采样点云已保存。")

if __name__ == "__main__":
    # 目标目录路径
    target_dir = "/mnt/e/IBS/dexmachina/arctic/output_meshes_mixer_use_01/timestep_0040"
    
    # 检查目录是否存在
    if os.path.exists(target_dir):
        process_hand_object_ibs(target_dir)
    else:
        print(f"错误: 目录不存在 {target_dir}")