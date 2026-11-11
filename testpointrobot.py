import yourdfpy
import trimesh
import numpy as np
import os
import torch # 导入 PyTorch
import datetime # 导入 datetime 模块用于时间戳

def get_robot_point_cloud(
    urdf_path: str,
    joint_angles: dict[str, float],
    base_pos: np.ndarray, # 此时 base_pos 和 base_quat 专指 base_link 的世界姿态
    base_quat: np.ndarray, # 此时 base_pos 和 base_quat 专指 base_link 的世界姿态
    device: torch.device = torch.device("cpu") # 新增 device 参数，默认为 CPU
) -> torch.Tensor | None: # 返回类型改为 torch.Tensor
    """
    从 URDF 文件和关节角度生成机械手的点云（兼容最新版 yourdfpy），
    并以 PyTorch 张量形式返回，适合 DifferentiableIBS 模块。
    该函数现在接受 base_link 的世界姿态作为输入。

    Args:
        urdf_path: URDF 文件路径。
        joint_angles: 关节角度字典 {关节名: 角度（弧度）}。
        base_pos: base_link 的世界位置 [x, y, z]。
        base_quat: base_link 的世界姿态四元数 [x, y, z,w]。
        device: PyTorch 张量将存储的设备 (e.g., torch.device("cpu") 或 torch.device("cuda")).

    Returns:
        点云 PyTorch 张量 (1, N, 3)，失败时返回 None。
    """
    try:
        # 1. 加载 URDF 并更新关节状态
        robot = yourdfpy.URDF.load(urdf_path)
        robot.update_cfg(joint_angles)  # 关键更新：设置关节角度
        print("URDF 加载成功，关节状态已更新。") # 减少输出
        print("base_pos:", base_pos) # 减少输出
        print("base_quat:", base_quat) # 减少输出
        # 2. 获取场景中所有网格顶点
        all_vertices = []
        for name, mesh in robot.scene.geometry.items():
            if isinstance(mesh, trimesh.Trimesh):
                # 获取网格的全局变换矩阵，这是从 URDF 'root' 链接到 mesh 的变换
                transform = robot.scene.graph.get(name)[0]
                # 转换顶点到 URDF 'root' 坐标系
                world_vertices = trimesh.transform_points(mesh.vertices, transform)
                all_vertices.append(world_vertices)

        if not all_vertices:
            print("警告: 未找到有效网格数据。")
            return None

        # --- 新增的逻辑：根据 base_link 的期望世界姿态计算 root 链接的实际世界姿态 ---

        # 2.1. 获取从 URDF 'root' 链接到 'base_link' 的当前变换
        # 这个变换是基于 'joint_angles' 计算出来的
        T_root_to_base_link_matrix = robot.get_transform('root', 'base_link')

        # 2.2. 将输入的 base_pos 和 base_quat 转换为 base_link 的期望世界齐次变换矩阵
        # 原始输入的 base_quat 是 [w, x, y, z]，trimesh.transformations.quaternion_matrix 期望 [x, y, z, w]
        base_quat_np = np.asarray(base_quat)


        
        T_world_to_base_link_desired = trimesh.transformations.quaternion_matrix(base_quat_np)
        T_world_to_base_link_desired[:3, 3] = base_pos

        # 2.3. 计算 'root' 链接在世界中应该处于的姿态 (T_world_to_root_required)
        # 关系： T_world_to_base_link_desired = T_world_to_root_required @ T_root_to_base_link_matrix
        # 所以： T_world_to_root_required = T_world_to_base_link_desired @ inv(T_root_to_base_link_matrix)
        
        # 注意：yourdfpy 返回的变换是父链接到子链接的变换。这里我们需要的是反向变换。
        # get_transform 返回的是 4x4 矩阵
        T_base_link_to_root_matrix = np.linalg.inv(T_root_to_base_link_matrix)
        
        # 这就是最终用于将整个点云从 URDF 'root' 坐标系转换到世界坐标系的变换
        final_root_transform_to_world = T_world_to_base_link_desired @ T_base_link_to_root_matrix

        # 3. 合并点云并应用计算出的 'root' 链接世界变换
        full_point_cloud = np.concatenate(all_vertices, axis=0)
        world_point_cloud_np = trimesh.transform_points(full_point_cloud, final_root_transform_to_world)

        # print(f"生成 NumPy 点云完成，点数: {len(world_point_cloud_np)}") # 减少输出

        # 4. 转换为 PyTorch 张量并添加批处理维度
        world_point_cloud_tensor = torch.tensor(world_point_cloud_np, dtype=torch.float32, device=device)
        world_point_cloud_tensor = world_point_cloud_tensor.unsqueeze(0) # 添加批处理维度 [1, N, 3]

        # print(f"转换为 PyTorch 张量，形状: {world_point_cloud_tensor.shape}") # 减少输出
        return world_point_cloud_tensor

    except Exception as e:
        print(f"错误: {e}")
        return None

# main 函数部分保持不变，现在它会正确工作，因为 get_robot_point_cloud 内部处理了
# base_pos/base_quat 是 base_link 姿态的情况。
if __name__ == '__main__':
    # --- 请替换为您的实际参数 ---

    # 1. InspireHand URDF 文件路径
    my_urdf_path = r"E:\IBS\dexmachina\dexmachina\assets\inspire_hand\handleft01151\urdf\left_xyz_first_mimic.urdf"

    # 2. 定义机械手所有关节的角度 (单位：弧度)
    # 这些角度会影响 root 到 base_link 的相对变换
    my_joint_angles = {
        'L_forearm_tx_link_joint': 0.0, # 设置为0
        'L_forearm_ty_link_joint': 0.0, # 设置为0
        'L_forearm_tz_link_joint': 0.0, # 设置为0
        'L_forearm_roll_link_joint': 0.0, # 设置为0
        'L_forearm_pitch_link_joint': 0.0, # 设置为0
        'L_forearm_yaw_link_joint': 0.0, # 设置为0
        'L_TH_J1': 0.5,
        'L_TH_J2': 0.3,
        'L_TH_J3': 0.2,
        'L_TH_J4': 0.4,
        'L_FF_J1': 0.8,
        'L_FF_J2': 0.5,
        'L_MF_J1': 0.8,
        'L_MF_J2': 0.5,
        'L_RF_J1': 0.8,
        'L_RF_J2': 0.5,
        'L_LF_J1': 0.8,
        'L_LF_J2': 0.5,
    }

    # 3. 定义机械手 base_link 在世界中的位姿
    # 期望 base_link 的原点在世界坐标系的 (0.5, 0.5, 0.5)
    my_base_pos = np.array([0.29193628 ,0.17849272 ,1.1393976 ])
    # 期望 base_link 的方向是单位四元数 [x, y, z, w]
    my_base_quat = np.array([ 0.9511283  , 0.2261257,  -0.1161185  , 0.17532428]) # (x, y, z, w) - 与 Isaac Gym 约定一致

    # 4. 设置 PyTorch 设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"IBS 计算将使用设备: {device}")

    if not os.path.exists(my_urdf_path):
        print(f"错误: 找不到 URDF 文件 '{my_urdf_path}'。请检查路径并确保文件存在。")
    else:
        # 调用函数获取 PyTorch 格式的点云
        hand_point_cloud_tensor = get_robot_point_cloud(
            urdf_path=my_urdf_path,
            joint_angles=my_joint_angles,
            base_pos=my_base_pos,
            base_quat=my_base_quat, # 现在直接传递 [x, y, z, w] 顺序
            device=device # 传递设备参数
        )

        if hand_point_cloud_tensor is not None:
            print(f"\n成功获取机械手点云（PyTorch 张量），形状：{hand_point_cloud_tensor.shape}，位于：{hand_point_cloud_tensor.device}")

            # 将 PyTorch 张量移到 CPU 并转换为 NumPy 数组
            hand_point_cloud_np = hand_point_cloud_tensor.squeeze(0).cpu().numpy()

            # 定义输出目录和文件名
            output_dir = "E:\\IBS\\dexmachina\\robot_point_clouds"
            os.makedirs(output_dir, exist_ok=True) # 确保目录存在
            
            # 使用时间戳创建唯一文件名
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            output_filename = os.path.join(output_dir, f"robot_point_cloud_{timestamp}.obj")

            # 将点云数据写入 .obj 文件
            with open(output_filename, 'w') as f:
                for vertex in hand_point_cloud_np:
                    # OBJ 文件的顶点格式为 "v x y z"
                    f.write(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}\n")
            
            print(f"点云已成功保存到: {output_filename}")

        else:
            print("未能生成机械手点云。")