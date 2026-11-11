

import numpy as np
import argparse
import os
'''
python tool/viewnpy.py --dir                                                                                 "E:\IBS\dexmachina\dexmachinatest12.22\inspire_hand\inspire-test_waffleiron40-340-s01-u01_B12000_hybrid_thres0.5_ho16_imi0.2_con2.0_bc0.2_ibsbeta25.0\nn\last_inspire_hand_ep_5000_rew__214.09793__eval"

'''
def view_npy_data(file_path):
    """
    加载并打印评估数据 .npy 文件的内容。
    假设文件是使用 np.save(filename, dictionary) 保存的。
    """
    if not os.path.exists(file_path):
        print(f"错误：文件不存在于路径：{file_path}")
        return

    try:
        # 加载数据，必须使用 allow_pickle=True 和 .item() 来加载保存的字典
        eval_data = np.load(file_path, allow_pickle=True).item()
        
    except Exception as e:
        print(f"加载文件时发生错误 {file_path}: {e}")
        return

    print("=" * 40)
    print(f"  评估数据文件：{file_path}")
    print("=" * 40)
    
    # 打印字典的键和每个数组的形状
    print("\n📦 数据内容 (Key 和 Shape):")
    for key, value in eval_data.items():
        if isinstance(value, np.ndarray):
            print(f"  - Key: **{key}**")
            print(f"    - Shape: {value.shape}")
            # 打印第一个时间步的数值作为示例
            print(f"    - 示例值 (第0步): {value[0]}")
        else:
            print(f"  - Key: **{key}** (非 NumPy 数组类型)")
            print(f"    - Value: {value}")
    
    # 额外分析一些关键数据
    print("\n📊 关键数据统计:")
    
    if 'obj_state' in eval_data and 'demo_state' in eval_data:
        obj_state = eval_data['obj_state']
        demo_state = eval_data['demo_state']
        
        # 假设 (T, N, D) -> T 是时间步，N 是环境数（评估时通常是 1），D 是维度
        print(f"  总时间步长 (T): {obj_state.shape[0]}")
        print(f"  环境数量 (N): {obj_state.shape[1]}")
        
        # 计算整个轨迹的平均 L2 距离（欧式距离）作为整体误差
        # 仅使用前 3 维（位置 Pos）计算
        pos_error = np.linalg.norm(obj_state[:, :, :3] - demo_state[:, :, :3], axis=-1)
        mean_pos_error = np.mean(pos_error)
        print(f"  平均对象位置误差 (L2): {mean_pos_error:.6f}")
        
    if 'pos_dist' in eval_data:
        pos_dist = eval_data['pos_dist']
        mean_pos_dist = np.mean(pos_dist)
        print(f"  平均 Pos Dist 奖励分量: {mean_pos_dist:.6f}")
        
    if 'rot_dist' in eval_data:
        rot_dist = eval_data['rot_dist']
        mean_rot_dist = np.mean(rot_dist)
        print(f"  平均 Rot Dist 奖励分量: {mean_rot_dist:.6f}")
        
    if 'arti_dist' in eval_data:
        arti_dist = eval_data['arti_dist']
        mean_arti_dist = np.mean(arti_dist)
        print(f"  平均 Arti Dist 奖励分量: {mean_arti_dist:.6f}")
    # also support alternative key naming like 'pos-dist'
    if 'pos-dist' in eval_data and 'pos_dist' not in eval_data:
        pos_dist = eval_data['pos-dist']
        mean_pos_dist = np.mean(pos_dist)
        print(f"  平均 Pos-Dist 奖励分量 (pos-dist): {mean_pos_dist:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="查看 RL 评估数据 .npy 文件的内容。")
    parser.add_argument('--file_path', type=str, help="要查看的 .npy 文件的完整路径。")
    parser.add_argument('--dir', type=str, help="要批量处理的目录，打印该目录下前20个 .npy 文件中 pos_dist 的均值。")
    
    args = parser.parse_args()
    # if directory provided, process up to 20 .npy files inside it
    if args.dir is not None:
        target_dir = args.dir
        if not os.path.isdir(target_dir):
            print(f"错误：目录不存在: {target_dir}")
        else:
            # collect npy files
            files = [f for f in os.listdir(target_dir) if f.endswith('.npy')]
            files = sorted(files)
            if len(files) == 0:
                print(f"目录中没有 .npy 文件: {target_dir}")
            else:
                print(f"在目录 {target_dir} 中找到 {len(files)} 个 .npy 文件，显示最多前 20 个的 pos_dist 平均值:\n")
                max_print = min(20, len(files))
                for i in range(max_print):
                    fname = files[i]
                    fpath = os.path.join(target_dir, fname)
                    try:
                        data = np.load(fpath, allow_pickle=True)
                        # handle both dict-like saved via np.save(..., dict) and raw arrays
                        if isinstance(data, np.lib.npyio.NpzFile):
                            # .npz archive
                            # try common keys
                            key_candidates = ['pos_dist', 'pos-dist', 'posDist']
                            found = False
                            for k in key_candidates:
                                if k in data:
                                    arr = data[k][:]
                                    mean_val = float(np.mean(arr))
                                    print(f"{fname}: mean({k}) = {mean_val:.6f}")
                                    found = True
                                    break
                            if not found:
                                # print available keys
                                print(f"{fname}: 未找到 pos_dist 键，可用键: {list(data.files)}")
                        else:
                            # plain .npy, possibly a saved dict via allow_pickle
                            try:
                                obj = data.item() if hasattr(data, 'item') else data
                            except Exception:
                                obj = data
                            mean_val = None
                            if isinstance(obj, dict):
                                key_candidates = ['pos_dist', 'pos-dist', 'posDist']
                                for k in key_candidates:
                                    if k in obj:
                                        arr = obj[k][:]
                                        mean_val = float(np.mean(arr))
                                        print(f"{fname}: mean({k}) = {mean_val:.6f}")
                                        break
                                if mean_val is None:
                                    print(f"{fname}: 未找到 pos_dist 键，文件键: {list(obj.keys())}")
                            else:
                                # obj is an array
                                try:
                                    mean_val = float(np.mean(obj))
                                    print(f"{fname}: mean(array) = {mean_val:.6f}")
                                except Exception as e:
                                    print(f"{fname}: 无法计算均值，类型={type(obj)} err={e}")
                    except Exception as e:
                        print(f"加载文件 {fname} 时出错: {e}")
    else:
        if args.file_path is None:
            print("请传入 --file_path 或 --dir 参数。")
        else:
            view_npy_data(args.file_path)