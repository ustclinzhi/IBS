import torch
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
import os
from typing import Optional # 导入 typing 模块中的 Optional

def get_two_link_box_point_cloud(
    bottom_stl_path: str,
    top_stl_path: str,
    box_root_pos: np.ndarray,
    box_root_quat: np.ndarray,
    joint_angle: float,
    device: torch.device = torch.device("cpu")
) -> Optional[torch.Tensor]:
    """
    根据双连杆盒子的几何与关节信息，计算其在世界坐标系中的点云，
    并以 PyTorch 张量形式返回。

    Args:
        bottom_stl_path (str): 'bottom' 连杆的 STL 文件路径。
        top_stl_path (str): 'top' 连杆的 STL 文件路径。
        box_root_pos (np.ndarray): 'bottom' 连杆根部的世界坐标 (x, y, z)。
        box_root_quat (np.ndarray): 'bottom' 连杆根部的世界姿态四元数 (x, y, z, w)。
        joint_angle (float): 关节旋转角度 (弧度)。
        device: PyTorch 张量将存储的设备 (e.g., torch.device("cpu") 或 torch.device("cuda")).

    Returns:
        torch.Tensor | None: 整个盒子的合并点云坐标 PyTorch 张量 (1, N, 3)。如果失败则返回 None。
    """
    all_world_vertices = []

    # --- 1. 处理 bottom 连杆 (根连杆) ---
    try:
        bottom_mesh = trimesh.load_mesh(bottom_stl_path)
    except Exception as e:
        print(f"Error: Unable to load bottom link STL file '{bottom_stl_path}'. {e}")
        return None

    # 将根连杆的局部顶点变换到世界坐标系
    try:
        # 确保四元数是 (x, y, z, w) 顺序
        bottom_rot_matrix = R.from_quat(box_root_quat).as_matrix()
    except ValueError as e:
        print(f"Error: Invalid quaternion format for 'bottom' link. {e}")
        return None
        
    bottom_world_vertices = bottom_mesh.vertices @ bottom_rot_matrix.T + box_root_pos
    all_world_vertices.append(bottom_world_vertices)

    # --- 2. 处理 top 连杆 ---
    try:
        top_mesh = trimesh.load_mesh(top_stl_path)
    except Exception as e:
        print(f"Error: Unable to load top link STL file '{top_stl_path}'. {e}")
        return None

    # 计算 top 连杆相对于 bottom 连杆的变换
    # URDF 定义: 关节绕 bottom 连杆坐标系的 Z 轴负方向旋转
    joint_axis = np.array([0.0, 0.0, -1.0])
    relative_rot_matrix = R.from_rotvec(joint_axis * joint_angle).as_matrix()
    
    # 首先，应用关节旋转，得到 top 连杆相对于 bottom 连杆的顶点坐标
    top_vertices_relative_to_bottom = top_mesh.vertices @ relative_rot_matrix.T

    # 然后，将这些相对坐标变换到世界坐标系
    top_world_vertices = top_vertices_relative_to_bottom @ bottom_rot_matrix.T + box_root_pos
    all_world_vertices.append(top_world_vertices)

    # --- 3. 合并点云 (NumPy 数组形式) ---
    final_point_cloud_np = np.concatenate(all_world_vertices, axis=0)
    print(f"NumPy point cloud generated successfully, total {len(final_point_cloud_np)} points.")

    # --- 4. 转换为 PyTorch 张量并添加批处理维度 ---
    # 使用 torch.tensor() 而不是 from_numpy() 避免潜在的共享内存问题，并明确指定 dtype 和 device
    final_point_cloud_tensor = torch.tensor(final_point_cloud_np, dtype=torch.float32, device=device)
    final_point_cloud_tensor = final_point_cloud_tensor.unsqueeze(0) # 添加批处理维度 [1, N, 3]

    print(f"Converted to PyTorch tensor, shape: {final_point_cloud_tensor.shape}, on: {final_point_cloud_tensor.device}")
    return final_point_cloud_tensor

def save_obj_from_npy_frame(
    npy_file_path: str,
    frame_idx: int,
    obj_bottom_stl_path: str,
    obj_top_stl_path: str,
    output_dir: str = "./output_obj_frames"
) -> Optional[torch.Tensor]:
    """
    从 .npy 文件中读取指定帧的物体位姿和关节角度，生成物体点云并保存为 .obj 文件。

    Args:
        npy_file_path (str): .npy 文件的路径。
        frame_idx (int): 要提取的帧索引。
        obj_bottom_stl_path (str): 'bottom' 连杆的 STL 文件路径。
        obj_top_stl_path (str): 'top' 连杆的 STL 文件路径。
        output_dir (str): 保存 .obj 文件的目录。

    Returns:
        torch.Tensor | None: 生成的物体点云张量，如果失败则返回 None。
    """
    if not os.path.exists(npy_file_path):
        print(f"Error: .npy file not found at '{npy_file_path}'.")
        return None
    if not os.path.exists(obj_bottom_stl_path):
        print(f"Error: Object bottom STL file not found at '{obj_bottom_stl_path}'.")
        return None
    if not os.path.exists(obj_top_stl_path):
        print(f"Error: Object top STL file not found at '{obj_top_stl_path}'.")
        return None

    try:
        # 加载 .npy 文件
        data = np.load(npy_file_path, allow_pickle=True).item()
        
        # 确保 'params' 键存在
        if 'params' not in data:
            print(f"Error: 'params' key not found in '{npy_file_path}'.")
            return None
        
        params = data['params']

        # 检查帧索引是否有效
        num_frames = params['obj_trans'].shape[0]
        if not (0 <= frame_idx < num_frames):
            print(f"Error: Frame index {frame_idx} is out of bounds. Max frame index is {num_frames - 1}.")
            return None

        # 提取指定帧的物体位姿和关节角度
        obj_root_pos = params['obj_trans'][frame_idx]
        obj_rot_euler = params['obj_rot'][frame_idx] # 假设是 Euler 角度 (roll, pitch, yaw)
        obj_joint_angle = params['obj_arti'][frame_idx]

        # 将 Euler 角度转换为四元数
        # 假设 obj_rot_euler 是 (roll, pitch, yaw) 顺序，单位为弧度
        # 如果是度数，需要 R.from_euler('xyz', obj_rot_euler, degrees=True)
        obj_root_quat = R.from_euler('xyz', obj_rot_euler, degrees=False).as_quat()

        print(f"\n--- Processing Frame {frame_idx} ---")
        print(f"Object Root Position: {obj_root_pos}")
        print(f"Object Root Quaternion (from Euler): {obj_root_quat}")
        print(f"Object Joint Angle: {np.degrees(obj_joint_angle):.2f} degrees ({obj_joint_angle:.4f} radians)")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {device}")

        # 获取物体点云
        obj_points = get_two_link_box_point_cloud(
            bottom_stl_path=obj_bottom_stl_path,
            top_stl_path=obj_top_stl_path,
            box_root_pos=obj_root_pos,
            box_root_quat=obj_root_quat,
            joint_angle=obj_joint_angle,
            device=device
        )

        if obj_points is None:
            print(f"Error: Could not generate object point cloud for frame {frame_idx}.")
            return None

        # 保存物体点云到 .obj 文件
        os.makedirs(output_dir, exist_ok=True)
        obj_points_np = obj_points.squeeze(0).cpu().numpy()
        obj_mesh = trimesh.points.PointCloud(obj_points_np)
        output_obj_path = os.path.join(output_dir, f"obj_frame_{frame_idx}.obj")
        obj_mesh.export(output_obj_path)
        print(f"Object point cloud for frame {frame_idx} saved to: {output_obj_path}")

        return obj_points

    except Exception as e:
        print(f"An error occurred while processing frame {frame_idx}: {e}")
        return None


if __name__ == '__main__':
    # --- 1. 定义物体 STL 文件路径 ---
    my_obj_bottom_stl_path = r"E:\IBS\dexmachina\dexmachina\assets\arctic\box\bottom_watertight_tiny.stl"
    my_obj_top_stl_path = r"E:\IBS\dexmachina\dexmachina\assets\arctic\box\top_watertight_tiny.stl"
    
    # --- 2. 定义 .npy 文件路径和要处理的帧 ---
    my_npy_file_path = r"E:\IBS\dexmachina\dexmachina\assets\arctic\processed\s01\box_use_01.npy" # 请确保这个路径是正确的，并且文件存在
    
    # 要提取的帧索引
    frame_to_extract = 80 # 例如，提取第 0 帧的数据
    output_directory_frames = fr"E:\IBS\dexmachina\arctic\output_meshes\s01_box_use_01_view1_meshes\obj_{frame_to_extract}.obj" # 单帧 OBJ 文件的输出目录

    # --- 3. 执行操作 ---
    # 检查 STL 文件是否存在
    if not os.path.exists(my_obj_bottom_stl_path):
        print(f"Error: Object bottom STL file not found at '{my_obj_bottom_stl_path}'. Please check the path.")
    elif not os.path.exists(my_obj_top_stl_path):
        print(f"Error: Object top STL file not found at '{my_obj_top_stl_path}'. Please check the path.")
    else:
        # 从 .npy 文件中提取特定帧的物体并保存为 OBJ
        print(f"\n--- Extracting and saving object from NPY file for frame {frame_to_extract} ---")
        obj_points_from_npy = save_obj_from_npy_frame(
            npy_file_path=my_npy_file_path,
            frame_idx=frame_to_extract,
            obj_bottom_stl_path=my_obj_bottom_stl_path,
            obj_top_stl_path=my_obj_top_stl_path,
            output_dir=output_directory_frames
        )
        if obj_points_from_npy is not None:
            print(f"Object for frame {frame_to_extract} successfully extracted and saved to '{output_directory_frames}'.")
        else:
            print(f"Failed to extract and save object for frame {frame_to_extract}.")

