import os
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import trimesh

URDF_PATH = Path(r"e:\IBS\dexmachina\dexmachina\assets\new_hand\left\mano_hand_left_multi.urdf")
MESH_DIR = Path(r"e:\IBS\dexmachina\dexmachina\assets\new_hand\left\meshes")
OUT_SNIPPET = Path(r"e:\IBS\dexmachina\dexmachina\assets\new_hand\left\origin_suggestions_assembly.txt")

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
    return joints

def get_link_centroid(mesh):
    """获取网格的质心"""
    return np.array(mesh.centroid)

def compute_relative_transform(parent_mesh, child_mesh):
    """
    对于装配坐标系下的STL文件，计算从父链接到子链接的相对变换
    这里我们假设在装配状态下，STL文件已经在正确的位置
    因此相对变换就是两个质心之间的差值
    """
    parent_centroid = get_link_centroid(parent_mesh)
    child_centroid = get_link_centroid(child_mesh)
    
    # 计算从父链接到子链接的相对位置
    relative_position = child_centroid - parent_centroid
    
    return relative_position

def write_snippets(snippets):
    with open(OUT_SNIPPET, 'w', encoding='utf-8') as f:
        for s in snippets:
            f.write(s+"\n\n")
    print("wrote", OUT_SNIPPET)

def main():
    joints = parse_urdf(URDF_PATH)
    # map link -> mesh name (assume mesh filename like link_joint_X.stl in URDF)
    # try to infer from URDF by searching <mesh filename="...">
    tree = ET.parse(str(URDF_PATH))
    root = tree.getroot()
    link_mesh = {}
    for link in root.findall('link'):
        ln = link.get('name')
        geo = link.find('.//geometry/mesh')
        if geo is not None:
            fname = os.path.basename(geo.get('filename'))
            link_mesh[ln] = fname

    # load meshes
    meshes = {}
    for ln,fname in link_mesh.items():
        m = load_mesh(fname)
        if m is None:
            print("missing mesh for", ln, fname)
        else:
            meshes[ln] = m

    snippets = []
    
    # 对于装配坐标系，我们需要计算相邻链接间的相对位置
    for j in joints:
        parent = j['parent']
        child = j['child']
        
        # 需要父和子网格都可用
        if parent not in meshes or child not in meshes:
            print(f"Missing mesh for joint {j['name']}, parent: {parent}, child: {child}")
            continue
            
        meshA = meshes[parent]
        meshB = meshes[child]
        
        # 计算相对变换
        relative_pos = compute_relative_transform(meshA, meshB)
        
        # 生成URDF片段
        snip = (f"<!-- joint {j['name']} -->\n"
                f"<joint name=\"{j['name']}\" type=\"revolute\">\n"
                f"  <parent link=\"{parent}\"/>\n"
                f"  <child link=\"{child}\"/>\n"
                f"  <origin xyz=\"{relative_pos[0]:.6f} {relative_pos[1]:.6f} {relative_pos[2]:.6f}\" rpy=\"0 0 0\"/>\n"
                f"  <axis xyz=\"0 0 1\"/>\n"  # 默认轴，实际可能需要根据具体设计调整
                f"</joint>")
        snippets.append(snip)
        
        # 计算子链接的视觉/碰撞原点（相对于其自身关节坐标系）
        child_centroid = get_link_centroid(meshB)
        # 在装配坐标系中，链接的原点应该在其质心相对于关节位置的偏移
        joint_pos = get_link_centroid(meshA) + relative_pos
        visual_origin = child_centroid - joint_pos
        
        snip2 = (f"<!-- visual/inertial for link {child} -->\n"
                 f"<link name=\"{child}\">\n"
                 f"  <!-- suggested visual/collision/inertial origin (rpy=0) -->\n"
                 f"  <origin xyz=\"{visual_origin[0]:.6f} {visual_origin[1]:.6f} {visual_origin[2]:.6f}\" rpy=\"0 0 0\" />\n"
                 f"</link>")
        snippets.append(snip2)

    write_snippets(snippets)
    print("Done. 为装配坐标系STL文件生成的URDF片段已写入。请检查并根据需要手动调整关节轴方向。")

if __name__ == "__main__":
    main()