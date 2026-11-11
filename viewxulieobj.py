import pyrender
import trimesh
import numpy as np
import os
import time
# conda activate test
# python viewxulieobj.py
# --- 配置参数 ---
# 替换为你的文件路径
demokpt_folder = r'E:\IBS\dexmachina\demo_kpts_save_l'  # demokpt 物体的文件夹
multi_obj_parent_folder = r'E:\IBS\dexmachina\arctic\output_meshes_notebook_use_01'  # 包含多个子文件夹的父文件夹

# 文件名前缀和后缀
demokpt_file_prefix = 'demo_kpts_left_step_'
multi_obj_file_suffix = '.obj'

# 动画播放速度 (秒)，你可以调整这个值来控制播放快慢
animation_speed = 0.06

# 新添加的点大小参数
point_size = 0.004  # 控制点的大小，根据需要调整这个值

# 定义点云渲染的材质
def create_point_material(color):
    return pyrender.MetallicRoughnessMaterial(
        baseColorFactor=color,
        metallicFactor=0.0,
        roughnessFactor=1.0,
        alphaMode='BLEND'
    )

current_colors = {
    'demokpt': [1.0, 0.0, 0.0, 1.0],  # demokpt 为红色
    'left_hand': [0.0, 1.0, 0.0, 1.0], # 假设lh为绿色，你可以根据需要调整
    'right_hand': [0.0, 0.0, 1.0, 1.0], # 假设rh为蓝色
    'object': [1.0, 1.0, 0.0, 1.0], # 假设object为黄色
}

# --- 代码开始 ---

# 获取所有子文件夹（即每一帧）
frame_folders = sorted([d for d in os.listdir(multi_obj_parent_folder)
                        if os.path.isdir(os.path.join(multi_obj_parent_folder, d))])
total_frames = len(frame_folders)

if total_frames == 0:
    print("错误：在父文件夹中未找到任何帧子文件夹。请检查路径。")
    exit()

print(f"检测到 {total_frames} 帧。")

# 创建一个 pyrender 场景
scene = pyrender.Scene()

# 创建一个 Viewer，用于动态显示场景
viewer = pyrender.Viewer(scene,
                         run_in_thread=True,
                         use_raymond_lighting=True)

# 循环遍历每一帧
for frame_idx, folder_name in enumerate(frame_folders):
    frame_number = frame_idx + 1


    print(f"正在加载第 {frame_number}/{total_frames} 帧...")

    # 锁定场景，以确保线程安全地修改
    viewer.render_lock.acquire()
    
    # 清空场景中所有之前的节点，只保留灯光等
    for node in list(scene.mesh_nodes):
        scene.remove_node(node)

    # --- 加载当前帧的物体 ---
    current_frame_meshes = {}
    
    # 1. 加载 demokpt 物体
    demokpt_filename = f"{demokpt_file_prefix}{(frame_number-29)}{multi_obj_file_suffix}"
    demokpt_path = os.path.join(demokpt_folder, demokpt_filename)
    
    if os.path.exists(demokpt_path):
        mesh = trimesh.load(demokpt_path)
        if isinstance(mesh, trimesh.points.PointCloud):
            current_frame_meshes['demokpt'] = mesh
        else:
            print(f"警告：文件 {demokpt_filename} 不是点云。跳过。")
    else:
        print(f"警告：demokpt 文件不存在 -> {demokpt_path}。跳过此帧的demokpt物体。")

    # 2. 加载另外三个物体
    current_frame_folder = os.path.join(multi_obj_parent_folder, folder_name)
    for obj_file in os.listdir(current_frame_folder):
        if obj_file.endswith(multi_obj_file_suffix) and not obj_file.startswith("ibs"):
            obj_path = os.path.join(current_frame_folder, obj_file)
            
            # 使用文件名（不含扩展名）作为物体的唯一标识
            obj_name = os.path.splitext(obj_file)[0]
            
            try:
                mesh = trimesh.load(obj_path)
                if isinstance(mesh, trimesh.points.PointCloud):
                    current_frame_meshes[obj_name] = mesh
                else:
                    print(f"警告：文件 {obj_file} 不是点云。跳过。")
            except Exception as e:
                print(f"加载文件失败 -> {obj_path}，错误：{e}")

    # 3. 将当前帧的所有物体添加到场景
    for obj_name, mesh in current_frame_meshes.items():
        # 从 trimesh 点云创建 pyrender 网格
        # 修改点大小的方法：将每个点转换为一个小球体
        sphere = trimesh.creation.uv_sphere(radius=point_size)
        sphere.visual.vertex_colors = current_colors.get(obj_name, [1.0, 0.0, 0.0, 1.0])
        
        # 为每个顶点创建一个球体
        transforms = np.tile(np.eye(4), (len(mesh.vertices), 1, 1))
        transforms[:, :3, 3] = mesh.vertices
        py_mesh = pyrender.Mesh.from_trimesh(sphere, poses=transforms)
        
        node = pyrender.Node(mesh=py_mesh)
        scene.add_node(node)

    # 释放锁定，允许 Viewer 渲染
    viewer.render_lock.release()

    # 控制动画速度
    time.sleep(animation_speed)
    
print("所有帧已加载，可视化窗口已启动。可以关闭窗口来结束程序。")

# 保持 Viewer 窗口打开，直到用户关闭它
while viewer.is_active:
    time.sleep(0.1)

viewer.close_external()