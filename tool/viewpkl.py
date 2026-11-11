#python tool/viewpkl.py E:\IBS\dexmachina\dexmachinatest\inspire_hand\inspire-example_box30-230-s01-u01_B4096_hybrid_thres0.6_ho16_imi0.3_con3.0_bc0.3_ibsbeta20.0\params\env.pkl
import os
import sys
from collections.abc import Mapping, Iterable
import argparse
import pickle
# 尝试导入 PyTorch，如果环境中没有则跳过
try:
    import torch
except ImportError:
    torch = None

# === 关键修复：处理 Windows 路径兼容性 (用于 pickle 加载可能存在的 PosixPath) ===
# 保留此修复以确保文件加载的兼容性
if sys.platform == "win32":
    import pathlib
    # 仅在 PosixPath 存在于 pathlib 中时才进行替换
    if hasattr(pathlib, 'PosixPath'):
        pathlib.PosixPath = pathlib.WindowsPath
# =======================================


def is_path_only_key(key):
    """检查键名是否只包含 'path' (不区分大小写)。"""
    if not isinstance(key, str):
        return False
    key_lower = key.lower()
    return 'path' in key_lower

def is_iterable_but_not_primitive(value):
    """判断一个值是否是可迭代的复杂结构，但排除字符串、张量和字典。"""
    if isinstance(value, str) or isinstance(value, Mapping):
        return False
    if torch and isinstance(value, torch.Tensor):
        return False
    
    return isinstance(value, Iterable)

# -------------------------------------------------------------------------------------
# 🎯 核心功能：递归查找并打印所有包含 'path' 关键字的键
# -------------------------------------------------------------------------------------
def recursively_print_path_keys(data, current_key_path=""):
    """
    递归遍历数据结构，查找并打印键名中包含 'path' 的所有项。
    """
    
    if isinstance(data, Mapping):
        for key, value in data.items():
            
            # 构造当前键的完整路径
            full_key_path = f"{current_key_path}.{key}" if current_key_path else str(key)
            
            # 检查当前键名是否包含 'path'
            if is_path_only_key(key):
                # 打印完整键路径和键值
                print("=" * 60)
                print(f"🔑 完整键路径: {full_key_path}")
                print(f"📝 键值 (类型: {type(value).__name__}):")
                
                # 尝试以可读性较好的方式打印键值
                if isinstance(value, (str, int, float, bool)) or (torch and isinstance(value, torch.Tensor)):
                     # 对于简单类型或 Tensor，直接打印
                    print(f"   {value}")
                elif isinstance(value, (list, tuple)):
                    # 对于列表或元组，打印前几个元素和长度
                    print(f"   (长度: {len(value)}) [ {str(value[:3])[:-1]}, ... ]")
                else:
                    # 对于其他复杂类型，打印类型信息
                    print(f"   <{type(value).__name__} object at {hex(id(value))}>")
                print("-" * 60)

            # 递归处理当前值
            recursively_print_path_keys(value, full_key_path)
    
    elif is_iterable_but_not_primitive(data):
        # 列表/元组：递归处理所有元素，使用索引作为路径的一部分
        for index, item in enumerate(data):
            list_key_path = f"{current_key_path}[{index}]"
            recursively_print_path_keys(item, list_key_path)

    # 其他类型 (Tensor, NumPy, str, int, float等)，直接返回 (不处理)
    return

# -------------------------------------------------------------------------------------

def process_pkl_data(file_path):
    """
    加载 .pkl 文件并执行基于键名包含 'path' 的查找和打印操作。
    """
    if not os.path.exists(file_path):
        print(f"错误：文件不存在于路径：{file_path}")
        return

    try:
        with open(file_path, 'rb') as f:
            # 禁用 'utf8' 错误，因为 pickle 文件可能是二进制的且包含各种对象
            original_data = pickle.load(f) 
        print(original_data['object_cfgs'])
        
    except Exception as e:
        print(f"加载文件时发生错误 {file_path}: {e}")
        return

    print("=" * 70)
    print(f" 🔎 正在分析文件：{file_path}")
    print(" 🎯 正在查找所有键名中包含 'path' (不区分大小写) 的键值对...")
    print("=" * 70)
    
    # 执行查找和打印
    recursively_print_path_keys(original_data)
    
    print("\n" + "=" * 70)
    print(" 🚀 查找完毕。")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="递归查找并打印 .pkl 文件中所有键名含有 'path' 的键值对。")
    parser.add_argument('file_path', type=str, help="要查看的 .pkl 文件的完整路径。")
    
    args = parser.parse_args()
    process_pkl_data(args.file_path)