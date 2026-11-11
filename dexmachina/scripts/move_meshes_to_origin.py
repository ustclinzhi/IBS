import os
import numpy as np
import trimesh
from pathlib import Path

MESH_DIR = Path(r"e:\IBS\dexmachina\dexmachina\assets\new_hand\left\meshes")
OUTPUT_DIR = Path(r"e:\IBS\dexmachina\dexmachina\assets\new_hand\left\meshes_centered")

def load_mesh(name):
    p = MESH_DIR / name
    if not p.exists():
        return None
    m = trimesh.load_mesh(str(p), force='mesh', skip_materials=True)
    # 如果坐标明显大（如 >1），假设是 mm，转换到 m
    if np.max(np.abs(m.bounds)) > 1.0:
        m.apply_scale(0.001)
    return m

def move_mesh_to_origin(mesh):
    """
    将网格移动到原点（使其质心位于原点）
    """
    centroid = mesh.centroid
    # 创建平移矩阵
    translation = -centroid
    # 应用平移
    mesh.apply_translation(translation)
    return mesh, translation

def main():
    # 创建输出目录
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # 获取所有STL文件
    stl_files = list(MESH_DIR.glob("*.stl"))
    print(f"Found {len(stl_files)} STL files")
    
    # 处理每个STL文件
    translations = {}
    for stl_file in stl_files:
        print(f"Processing {stl_file.name}")
        try:
            # 加载网格
            mesh = load_mesh(stl_file.name)
            if mesh is None:
                print(f"Failed to load {stl_file.name}")
                continue
            
            # 移动到原点
            centered_mesh, translation = move_mesh_to_origin(mesh)
            translations[stl_file.name] = translation
            
            # 保存到新位置
            output_path = OUTPUT_DIR / stl_file.name
            centered_mesh.export(str(output_path))
            print(f"Saved centered mesh to {output_path}")
            print(f"  Translation applied: {translation}")
            
        except Exception as e:
            print(f"Error processing {stl_file.name}: {e}")
    
    # 保存变换信息
    translation_file = OUTPUT_DIR / "translations.txt"
    with open(translation_file, 'w') as f:
        for mesh_name, translation in translations.items():
            f.write(f"{mesh_name}: {translation[0]:.6f}, {translation[1]:.6f}, {translation[2]:.6f}\n")
    
    print(f"Translations saved to {translation_file}")
    print("Done!")

if __name__ == "__main__":
    main()