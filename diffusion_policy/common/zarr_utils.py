#!/usr/bin/env python3
"""
将pickle格式的数据转换为更稳定的zarr格式
解决NumPy版本兼容性问题
"""

import json
import pickle
import sys
import traceback
from pathlib import Path
from typing import Any, Dict

import numpy as np
import zarr


def create_numpy_compatibility_patch():
    """创建NumPy兼容性补丁，临时解决加载问题"""
    import sys

    # 修补numpy.core.multiarray缺失的类型属性
    try:
        import numpy as np
        from numpy.core import multiarray

        missing_types = {
            "flexible": np.flexible,
            "number": np.number,
            "integer": np.integer,
            "signedinteger": np.signedinteger,
            "unsignedinteger": np.unsignedinteger,
            "inexact": np.inexact,
            "floating": np.floating,
            "complexfloating": np.complexfloating,
            "character": np.character,
            "bool_": np.bool_,
            "void": np.void,
        }

        for attr_name, attr_value in missing_types.items():
            if not hasattr(multiarray, attr_name):
                setattr(multiarray, attr_name, attr_value)

    except Exception as e:
        print(f"警告：NumPy兼容性补丁失败: {e}")

    # 创建numpy._core映射
    if "numpy._core" not in sys.modules:
        try:
            import numpy.core as _core

            sys.modules["numpy._core"] = _core

            # 映射子模块
            if hasattr(_core, "multiarray"):
                sys.modules["numpy._core.multiarray"] = _core.multiarray
            if hasattr(_core, "numeric"):
                sys.modules["numpy._core.numeric"] = _core.numeric
            if hasattr(_core, "umath"):
                sys.modules["numpy._core.umath"] = _core.umath

        except ImportError:
            pass


def safe_load_pickle(filepath: str) -> Any:
    """安全加载pickle文件"""
    try:
        with open(filepath, "rb") as f:
            return pickle.load(f, fix_imports=True, encoding="latin1")
    except Exception as e:
        print(f"标准加载失败 {filepath}: {e}")
        try:
            with open(filepath, "rb") as f:
                return pickle.load(f)
        except Exception as e2:
            print(f"备用加载也失败 {filepath}: {e2}")
            return None


def convert_pickle_to_zarr(input_dir: str, output_file: str) -> bool:
    """
    将pickle格式的数据转换为zarr格式(适配Diffusion Policy)

    Args:
        input_dir: 输入目录路径
        output_file: 输出zarr文件路径

    Returns:
        转换是否成功
    """
    try:
        print(f"开始转换: {input_dir} -> {output_file}")

        # 1. 扫描数据结构
        input_path = Path(input_dir)
        if not input_path.exists():
            print(f"错误：输入目录不存在: {input_dir}")
            return False

        episodes = sorted(
            [d for d in input_path.iterdir() if d.is_dir()],
            key=lambda x: int(x.name.split("_")[-1]),
        )

        if not episodes:
            print("错误：没有找到episode目录")
            return False

        # 2. 分析数据结构
        first_episode = episodes[0]
        data_types = [d.name for d in first_episode.iterdir() if d.is_dir()]
        # 不排除任何数据类型，全部转换
        print(f"原始数据类型: {data_types}")

        print(f"找到 {len(episodes)} 个episodes")
        print(f"数据类型: {data_types}")

        # 3. 创建zarr存储
        store = zarr.DirectoryStore(output_file)
        root = zarr.group(store=store, overwrite=True)

        # 创建data和meta子组
        data_group = root.create_group("data")
        meta_group = root.create_group("meta")

        # 4. 转换每种数据类型
        converted_data = {}
        episode_lengths = []

        for episode_idx, episode_dir in enumerate(episodes):
            print(f"处理episode {episode_idx}/{len(episodes)}: {episode_dir.name}")

            episode_data = {}

            for data_type in data_types:
                data_path = episode_dir / data_type
                if not data_path.exists():
                    continue

                # 加载该类型的所有文件
                files = sorted(data_path.glob("*.pkl"), key=lambda x: int(x.stem))

                type_data = []
                for file_path in files:
                    data = safe_load_pickle(str(file_path))
                    if data is not None:
                        type_data.append(data)
                    else:
                        print(f"跳过文件: {file_path}")

                if type_data:
                    # 转换为numpy数组
                    if isinstance(type_data[0], np.ndarray):
                        episode_data[data_type] = np.stack(type_data)
                    else:
                        episode_data[data_type] = np.array(type_data)

            # 记录episode长度
            if episode_data:
                episode_length = len(list(episode_data.values())[0])
                episode_lengths.append(episode_length)

                # 添加到总数据中
                for data_type, data_array in episode_data.items():
                    if data_type not in converted_data:
                        converted_data[data_type] = []
                    converted_data[data_type].append(data_array)

        # 5. 合并所有episodes的数据
        print("合并数据...")
        for data_type, episode_list in converted_data.items():
            if episode_list:
                # 拼接所有episodes
                combined_data = np.concatenate(episode_list, axis=0)

                # 保存到data组
                print(f"保存 {data_type}: {combined_data.shape}")
                data_group.create_dataset(
                    data_type,
                    data=combined_data,
                    chunks=combined_data.shape,  # 自动选择chunk大小
                    compression=None,  # 使用压缩
                    dtype=combined_data.dtype,
                )

        # 6. 保存元数据到meta组
        episode_lengths_array = np.array(episode_lengths, dtype=np.int64)
        episode_ends_array = np.cumsum(episode_lengths_array)

        # 保存各种元数据
        meta_group.create_dataset("episode_lengths", data=episode_lengths_array)
        meta_group.create_dataset("episode_ends", data=episode_ends_array)
        meta_group.create_dataset(
            "n_episodes", data=np.array(len(episodes), dtype=np.int64)
        )
        meta_group.create_dataset(
            "total_steps", data=np.array(sum(episode_lengths), dtype=np.int64)
        )

        # 保存数据类型列表为属性
        meta_group.attrs["data_types"] = list(converted_data.keys())

        # 为兼容性，也在root级别保存一份元数据
        metadata = {
            "n_episodes": len(episodes),
            "episode_lengths": episode_lengths,
            "episode_ends": episode_ends_array.tolist(),
            "data_types": list(converted_data.keys()),
            "total_steps": sum(episode_lengths),
        }
        root.attrs["metadata"] = json.dumps(metadata)

        print("✅ 转换完成！")
        print(f"总episodes: {metadata['n_episodes']}")
        print(f"总步数: {metadata['total_steps']}")
        print(f"数据类型: {metadata['data_types']}")
        print(f"输出文件: {output_file}")

        return True

    except Exception as e:
        print(f"❌ 转换失败: {e}")
        traceback.print_exc()
        return False


def load_zarr_data(zarr_file: str) -> Dict[str, Any]:
    """
    加载zarr格式的数据

    Args:
        zarr_file: zarr文件路径

    Returns:
        包含数据和元数据的字典
    """
    try:
        store = zarr.DirectoryStore(zarr_file)
        root = zarr.group(store=store, mode="r")

        # 加载元数据
        metadata = json.loads(root.attrs["metadata"])

        # 加载数据
        data = {}
        for key in root.keys():
            data[key] = root[key]

        return {"data": data, "metadata": metadata}

    except Exception as e:
        print(f"加载zarr数据失败: {e}")
        return None


def main():
    """主函数"""
    if len(sys.argv) != 3:
        print("用法: python convert_pickle_to_zarr.py <input_dir> <output_zarr>")
        print(
            "示例: python convert_pickle_to_zarr.py ./data/continuous_data_dev ./data/continuous_data_dev.zarr"
        )
        return

    input_dir = sys.argv[1]
    output_file = sys.argv[2]

    # 应用兼容性补丁
    create_numpy_compatibility_patch()

    # 执行转换
    success = convert_pickle_to_zarr(input_dir, output_file)

    if success:
        print("\n🎉 数据转换成功！")
        print("现在你可以使用zarr格式的数据，避免pickle兼容性问题")
        print("加载示例:")
        print("import zarr")
        print(f"data = zarr.open('{output_file}', mode='r')")
    else:
        print("\n❌ 数据转换失败")
        sys.exit(1)


if __name__ == "__main__":
    main()
