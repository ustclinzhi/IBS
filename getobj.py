import numpy as np

def extract_frame_and_save_obj(npy_file_path, output_obj_file_path):
    """
    从 (200, 18, 3) 形状的 .npy 文件中提取第一帧的 18 个点，
    并将其作为点云保存为 .obj 文件。

    Args:
        npy_file_path (str): 输入 .npy 文件的路径。
        output_obj_file_path (str): 输出 .obj 文件的路径。
    """
    try:
        # 1. 加载 .npy 文件
        data = np.load(npy_file_path)

        # 检查数据形状是否符合预期
        if data.shape != (200, 18, 3):
            print(f"警告: 输入 .npy 文件形状为 {data.shape}，而非预期的 (200, 18, 3)。")


        # 2. 提取第一帧的 18 个点 (索引为 0)
        # 结果将是一个 (18, 3) 的 NumPy 数组
        first_frame_points = data[49]

        # 3. 将点写入 .obj 文件
        with open(output_obj_file_path, 'w') as f:
            # OBJ 文件通常以注释开始
            f.write("# OBJ File generated from Python script\n")
            f.write("# Extracted first frame (18 points) from .npy file\n")

            # 遍历每一个点并写入为顶点 'v'
            for i, point in enumerate(first_frame_points):
                # OBJ 格式：v x y z
                f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
                # 打印进度（可选）
                # print(f"写入点 {i+1}: {point}")

        print(f"成功从 '{npy_file_path}' 提取第一帧并保存为 '{output_obj_file_path}'")
        print(f"共写入 {len(first_frame_points)} 个点。")

    except FileNotFoundError:
        print(f"错误: 找不到文件 '{npy_file_path}'。请检查路径是否正确。")
    except ValueError as e:
        print(f"数据格式错误: {e}")
    except Exception as e:
        print(f"发生未知错误: {e}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 请将 'your_data.npy' 替换为你实际的 .npy 文件路径
   
    input_npy_file = fr"E:\IBS\dexmachina\arctic\output_meshes\demoibs_left.npy"
    # 定义输出的 .obj 文件名
    output_obj_file = fr"E:\IBS\dexmachina\arctic\output_meshes\demoibs_left_50.obj"

    
    

    # 调用函数来执行提取和保存操作
    extract_frame_and_save_obj(input_npy_file, output_obj_file)