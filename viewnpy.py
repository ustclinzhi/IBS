import numpy as np
import matplotlib.pyplot as plt

# 设置中文字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass

def read_pos_dist_histogram(file_path):
    """
    读取 pos_dist 并显示直方图
    """
    print("=" * 60)
    print("读取 pos_dist 数据并生成直方图")
    print("=" * 60)
    
    try:
        # 读取数据
        print("\n正在读取文件...")
        data = np.load(file_path, allow_pickle=True)
        print(f"✓ 文件读取成功")
        
        # 转换为Python对象
        obj = data.tolist()
        
        # 如果是字典，提取 pos_dist
        if isinstance(obj, dict):
            print(f"字典的键: {list(obj.keys())}")
            
            if 'pos_dist' in obj:
                pos_dist = obj['pos_dist']
                print(f"✓ 找到 'pos_dist' 键")
                
                # 确保是numpy数组
                if not isinstance(pos_dist, np.ndarray):
                    pos_dist = np.array(pos_dist)
                
                print(f"  pos_dist 形状: {pos_dist.shape}")
                print(f"  pos_dist 数据类型: {pos_dist.dtype}")
                
                # 展平数据
                flat_data = pos_dist.flatten()
                print(f"  展平后数据量: {len(flat_data)}")
                
                # 检查是否是数值数据
                if np.issubdtype(pos_dist.dtype, np.number):
                    # 统计信息
                    print("\n" + "=" * 60)
                    print("统计信息:")
                    print("=" * 60)
                    print(f"最小值:     {flat_data.min():.6f}")
                    print(f"最大值:     {flat_data.max():.6f}")
                    print(f"平均值:     {flat_data.mean():.6f}")
                    print(f"中位数:     {np.median(flat_data):.6f}")
                    print(f"标准差:     {flat_data.std():.6f}")
                    print(f"90%分位数:  {np.percentile(flat_data, 90):.6f}")
                    
                    # 绘制直方图
                    print("\n正在生成直方图...")
                    plt.figure(figsize=(10, 6))
                    
                    # 绘制直方图
                    n, bins, patches = plt.hist(flat_data, bins=50, edgecolor='black', 
                                               alpha=0.7, color='steelblue')
                    
                    # 添加统计线
                    mean_val = flat_data.mean()
                    median_val = np.median(flat_data)
                    percentile_90 = np.percentile(flat_data, 90)
                    
                    plt.axvline(mean_val, color='red', linestyle='-', 
                               linewidth=2, label=f'均值: {mean_val:.4f}')
                    plt.axvline(median_val, color='orange', linestyle='--', 
                               linewidth=2, label=f'中位数: {median_val:.4f}')
                    plt.axvline(percentile_90, color='green', linestyle=':', 
                               linewidth=2, label=f'90%分位数: {percentile_90:.4f}')
                    
                    # 添加标签和标题
                    plt.xlabel('pos_dist 值', fontsize=12)
                    plt.ylabel('频数', fontsize=12)
                    plt.title('pos_dist 分布直方图', fontsize=14, fontweight='bold')
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    
                    # 显示图表
                    plt.tight_layout()
                    plt.show()
                    
                    print("✓ 直方图显示完成")
                    
                    # 可选：保存图片
                    # plt.savefig('pos_dist_histogram.png', dpi=300, bbox_inches='tight')
                    # print("✓ 图片已保存为: pos_dist_histogram.png")
                    
                    return pos_dist
                else:
                    print(f"\n⚠ pos_dist 不是数值类型: {pos_dist.dtype}")
                    print(f"  前5个数据: {flat_data[:5]}")
                    return pos_dist
            else:
                print(f"\n✗ 错误: 数据中没有 'pos_dist' 键")
                print(f"  可用的键: {list(obj.keys())}")
                return None
        else:
            print(f"\n数据不是字典，类型为: {type(obj)}")
            print(f"数据内容: {obj}")
            return obj
            
    except FileNotFoundError:
        print(f"\n✗ 错误: 文件未找到")
        print(f"  路径: {file_path}")
        return None
    except Exception as e:
        print(f"\n✗ 读取错误: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    # 您的文件路径
    file_path = r"E:\IBS\dexmachina\dexmachinatest\inspire_hand\inspire-example_box30-230-s01-u01_B4096_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3_ibsbeta20.0\nn\last_inspire_hand_ep_5000_rew_137.78058_eval\eval_ep7.npy"
    
    # 执行读取并显示直方图
    pos_dist = read_pos_dist_histogram(file_path)
    
    if pos_dist is not None:
        print("\n" + "=" * 60)
        print("✓ 数据读取完成")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("✗ 数据读取失败")
        print("=" * 60)