import numpy as np
from typing import Any, Dict, Union, Optional

def print_dict_keys_and_dtypes(data: Union[Dict[str, Any], np.ndarray, Any], prefix: str = "") -> None:
    """
    递归打印字典（或numpy数组包装的字典）的所有键、子键及对应值的数据类型

    参数:
        data: 输入数据（可以是dict、numpy数组或单个对象）
        prefix: 键的前缀（用于嵌套结构的层级显示）
    """
    # 处理numpy数组（核心修复：支持0维、1维、多维数组）
    if isinstance(data, np.ndarray):
        # 打印数组基本信息
        array_info = f"numpy数组 -> 形状: {data.shape}, 数据类型: {data.dtype}"
        print(f"{prefix}{array_info}")
        
        # 处理0维数组（标量数组）：直接取item()获取内部对象
        if data.ndim == 0:
            if data.size > 0:
                inner_data = data.item()  # 提取0维数组中的单个对象
                print_dict_keys_and_dtypes(inner_data, prefix=f"{prefix}  -> 内部对象: ")
            else:
                print(f"{prefix}  -> 空0维数组")
            return
        
        # 处理1维及多维数组：遍历所有元素
        for idx, elem in np.ndenumerate(data):  # ndenumerate支持多维数组索引
            elem_prefix = f"{prefix}  数组{idx}."  # 多维索引显示为 (i,j,k)
            if isinstance(elem, dict):
                print_dict_keys_and_dtypes(elem, prefix=elem_prefix)
            else:
                print(f"{elem_prefix} -> 数据类型: {type(elem).__name__} (非字典类型)")
        return
    
    # 处理字典类型（核心逻辑）
    if isinstance(data, dict):
        for key, value in data.items():
            current_prefix = f"{prefix}{key}" if prefix else key
            
            # 判断值的类型
            if isinstance(value, dict):
                # 子字典，递归处理
                print(f"{current_prefix} -> 数据类型: dict (嵌套字典)")
                print_dict_keys_and_dtypes(value, prefix=f"{current_prefix}.")
            elif isinstance(value, np.ndarray):
                # numpy数组，显示形状和dtype
                print(f"{current_prefix} -> 数据类型: numpy.ndarray (形状: {value.shape}, dtype: {value.dtype})")
            elif isinstance(value, (list, tuple)):
                # 列表/元组，显示长度和元素类型（只看第一个元素的类型）
                elem_type = type(value[0]).__name__ if len(value) > 0 else "空"
                print(f"{current_prefix} -> 数据类型: {type(value).__name__} (长度: {len(value)}, 元素类型: {elem_type})")
            elif value is None:
                print(f"{current_prefix} -> 数据类型: NoneType")
            else:
                # 其他基础类型（int, float, str, bool等）
                print(f"{current_prefix} -> 数据类型: {type(value).__name__}")
    else:
        # 既不是字典也不是numpy数组的情况
        print(f"{prefix} -> 数据类型: {type(data).__name__} (非字典类型)")

def extract_actual_data(loaded_data: np.ndarray) -> Any:
    """
    从numpy数组中提取实际数据（处理0维、1维数组的情况）
    """
    # 0维数组：直接返回内部对象
    if loaded_data.ndim == 0:
        return loaded_data.item() if loaded_data.size > 0 else None
    # 1维数组且只有一个元素：返回该元素（可能是字典）
    elif loaded_data.ndim == 1 and len(loaded_data) == 1:
        return loaded_data[0]
    # 其他情况：返回原数组
    return loaded_data

def main():
    import sys
    # 检查命令行参数
    if len(sys.argv) != 2:
        print("使用方法: python tool/viewnpygeshi.py E:\IBS\dexmachina\arctic\outputs\processed_verts\seqs\s01\box_use_01.npy")
        print("示例: python viewnpygeshi.py data.npy")
        sys.exit(1)
    
    npy_path = sys.argv[1]
    
    try:
        # 加载npy文件（allow_pickle=True必须，否则无法加载字典对象）
        loaded_data = np.load(npy_path, allow_pickle=True)
        print(f"成功加载文件: {npy_path}")
        print("=" * 60)
        
        # 提取实际数据（处理0维数组等情况）
        actual_data = extract_actual_data(loaded_data)
        
        # 打印完整的数据结构
        print("数据完整结构:")
        print("-" * 40)
        print_dict_keys_and_dtypes(actual_data)
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 '{npy_path}'")
    except ValueError as e:
        print(f"错误: 提取数组数据时出错 - {str(e)}")
        print("提示: 该文件可能不是存储字典/对象的npy文件")
    except Exception as e:
        print(f"错误: 加载或解析文件时发生异常 - {str(e)}")
        print(f"异常类型: {type(e).__name__}")

if __name__ == "__main__":
    main()