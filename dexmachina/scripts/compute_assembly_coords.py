import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh
from scipy.spatial import cKDTree

URDF_PATH = Path(r"e:\IBS\dexmachina\dexmachina\assets\new_hand\left\mano_hand_left_multi.urdf")
MESH_DIR = Path(r"e:\IBS\dexmachina\dexmachina\assets\new_hand\left\meshes")
OUT_URDF = Path(r"e:\IBS\dexmachina\dexmachina\assets\new_hand\left\mano_hand_left_multi_corrected.urdf")

def load_mesh(name):
    p = MESH_DIR / name
    if not p.exists():
        return None
    m = trimesh.load_mesh(str(p), force='mesh', skip_materials=True)
    # 如果坐标明显大（如 >1），假设是 mm，转换到 m
    if np.max(np.abs(m.bounds)) > 1.0:
        m.apply_scale(0.001)
    return m

def parse_urdf(urdf_path):
    tree = ET.parse(str(urdf_path))
    root = tree.getroot()
    joints = []
    for j in root.findall('joint'):
        name = j.get('name')
        parent = j.find('parent').get('link')
        child = j.find('child').get('link')
        origin = j.find('origin')
        if origin is None:
            xyz = [0.0,0.0,0.0]; rpy=[0.0,0.0,0.0]
        else:
            xyz = [float(x) for x in origin.get('xyz','0 0 0').split()]
            rpy = [float(x) for x in origin.get('rpy','0 0 0').split()]
        axis = j.find('axis')
        axis_v = np.array([0.0,0.0,1.0]) if axis is None else np.array([float(x) for x in axis.get('xyz').split()])
        joints.append({'name':name,'parent':parent,'child':child,'origin_xyz':xyz,'origin_rpy':rpy,'axis':axis_v})
    links = []
    for link in root.findall('link'):
        name = link.get('name')
        links.append({'name': name})
    return joints, links

def find_closest_points(meshA, meshB, samples=2000):
    """
    找到两个网格之间最近的点对
    """
    ptsA = meshA.sample(samples)
    ptsB = meshB.sample(samples)
    treeB = cKDTree(ptsB)
    dists, idxs = treeB.query(ptsA, k=1)
    min_idx = int(np.argmin(dists))
    pa = ptsA[min_idx]
    pb = ptsB[int(idxs[min_idx])]
    return pa, pb

def compute_joint_origin(parent_mesh, child_mesh):
    """
    基于装配坐标系计算关节原点
    关节原点应该是两个网格最接近的点
    """
    # 找到两个网格之间最接近的点
    closest_point_parent, closest_point_child = find_closest_points(parent_mesh, child_mesh)
    
    # 关节原点应该是父网格上的最近点（在父网格坐标系中）
    # 但由于所有网格都在装配坐标系中，我们可以直接使用该点
    joint_origin = closest_point_parent
    
    return joint_origin, closest_point_parent, closest_point_child

def compute_link_visual_origin(link_mesh, joint_origin):
    """
    计算link的visual origin
    应该是网格质心相对于关节原点的位置
    """
    centroid = np.array(link_mesh.centroid)
    visual_origin = centroid - joint_origin
    return visual_origin

def main():
    joints, links = parse_urdf(URDF_PATH)
    
    # 映射link名称到网格文件名
    tree = ET.parse(str(URDF_PATH))
    root = tree.getroot()
    link_mesh_filenames = {}
    for link in root.findall('link'):
        ln = link.get('name')
        geo = link.find('.//geometry/mesh')
        if geo is not None:
            fname = os.path.basename(geo.get('filename'))
            link_mesh_filenames[ln] = fname

    # 加载所有网格
    meshes = {}
    for ln, fname in link_mesh_filenames.items():
        m = load_mesh(fname)
        if m is not None:
            meshes[ln] = m
            print(f"Loaded mesh for {ln}: {fname}")

    # 创建新的URDF内容
    # 先读取原始URDF
    with open(URDF_PATH, 'r', encoding='utf-8') as f:
        urdf_lines = f.readlines()

    # 处理joint部分
    new_urdf_lines = []
    i = 0
    while i < len(urdf_lines):
        line = urdf_lines[i]
        if '<joint name="' in line and 'type="revolute"' in line:
            # 提取joint名称
            joint_name = line.split('"')[1]
            
            # 找到对应的joint数据
            joint_data = None
            for j in joints:
                if j['name'] == joint_name:
                    joint_data = j
                    break
            
            if joint_data and joint_data['parent'] in meshes and joint_data['child'] in meshes:
                parent_mesh = meshes[joint_data['parent']]
                child_mesh = meshes[joint_data['child']]
                
                # 计算新的joint origin
                joint_origin, _, _ = compute_joint_origin(parent_mesh, child_mesh)
                
                # 添加joint定义直到origin行
                new_urdf_lines.append(line)
                i += 1
                while i < len(urdf_lines) and '<origin' not in urdf_lines[i]:
                    new_urdf_lines.append(urdf_lines[i])
                    i += 1
                
                # 替换origin行
                origin_line = f'    <origin xyz="{joint_origin[0]:.6f} {joint_origin[1]:.6f} {joint_origin[2]:.6f}" rpy="0 0 0" />\n'
                new_urdf_lines.append(origin_line)
                i += 1
                
                # 继续添加剩余的joint定义
                while i < len(urdf_lines) and '</joint>' not in urdf_lines[i]:
                    # 跳过原来的origin行
                    if '<origin' not in urdf_lines[i]:
                        new_urdf_lines.append(urdf_lines[i])
                    i += 1
                new_urdf_lines.append(urdf_lines[i])  # 添加</joint>行
            else:
                new_urdf_lines.append(line)
        elif '<link name="' in line and 'inertial' in ''.join(urdf_lines[i:i+10]):
            # 处理link inertial origin
            link_name = line.split('"')[1]
            
            if link_name in meshes:
                # 找到相关的joint来确定joint_origin
                joint_data = None
                for j in joints:
                    if j['child'] == link_name and j['parent'] in meshes:
                        joint_data = j
                        break
                
                if joint_data:
                    parent_mesh = meshes[joint_data['parent']]
                    child_mesh = meshes[joint_data['child']]
                    
                    # 计算joint origin
                    joint_origin, _, _ = compute_joint_origin(parent_mesh, child_mesh)
                    
                    # 计算link的visual origin
                    link_mesh = meshes[link_name]
                    visual_origin = compute_link_visual_origin(link_mesh, joint_origin)
                    
                    # 添加link定义直到inertial origin行
                    new_urdf_lines.append(line)
                    i += 1
                    while i < len(urdf_lines) and '<origin' not in urdf_lines[i]:
                        new_urdf_lines.append(urdf_lines[i])
                        i += 1
                    
                    # 替换inertial origin行
                    inertial_line = f'      <origin xyz="{visual_origin[0]:.6f} {visual_origin[1]:.6f} {visual_origin[2]:.6f}" rpy="0 0 0" />\n'
                    new_urdf_lines.append(inertial_line)
                    i += 1
                    
                    # 继续添加剩余的link定义
                    while i < len(urdf_lines) and '</link>' not in urdf_lines[i]:
                        # 跳过原来的inertial origin行
                        if '<origin' not in urdf_lines[i] or 'inertial' not in urdf_lines[i-1] and 'inertial' not in urdf_lines[i-2]:
                            new_urdf_lines.append(urdf_lines[i])
                        i += 1
                    new_urdf_lines.append(urdf_lines[i])  # 添加</link>行
                else:
                    new_urdf_lines.append(line)
            else:
                new_urdf_lines.append(line)
        else:
            new_urdf_lines.append(line)
        i += 1

    # 写入新的URDF文件
    with open(OUT_URDF, 'w', encoding='utf-8') as f:
        f.writelines(new_urdf_lines)
    
    print(f"Corrected URDF written to {OUT_URDF}")

if __name__ == "__main__":
    main()