import multiprocessing
import os
import pathlib
from typing import Dict, Optional, Sequence, Union

import numcodecs
import numpy as np
import zarr
from tqdm import tqdm

from diffusion_policy.codecs.imagecodecs_numcodecs import register_codecs
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.common.replay_buffer import ReplayBuffer

# 只在需要时注册编解码器，避免重复注册警告
register_codecs(verbose=False)


def my_real_data_to_replay_buffer(
    dataset_path: str,  # * zarr 文件路径
    out_store: Optional[zarr.ABSStore] = None,  #
    out_resolutions: Union[
        None, tuple, Dict[str, tuple]
    ] = None,  # * 'camera': (width, height)
    lowdim_keys: Optional[Sequence[str]] = None,  # * [robot_eef_pose, action]
    image_keys: Optional[Sequence[str]] = None,  # * [camera]
    lowdim_compressor: Optional[numcodecs.abc.Codec] = None,
    image_compressor: Optional[numcodecs.abc.Codec] = None,
    n_decoding_threads: int = multiprocessing.cpu_count(),
    n_encoding_threads: int = multiprocessing.cpu_count(),
    max_inflight_tasks: int = multiprocessing.cpu_count() * 5,
    verify_read: bool = True,
):
    input = pathlib.Path(os.path.expanduser(dataset_path))
    in_zarr_path = input.joinpath("replay_buffer.zarr")
    # in_video_dir = input.joinpath('videos') 我们的数据直接把bgr图片保存在data内
    assert in_zarr_path.is_dir()
    in_replay_buffer = ReplayBuffer.create_from_path(
        str(in_zarr_path.absolute()), mode="r"
    )

    # 将lowdim 数据放在一个chunk里面
    chunks_map = dict()
    compressor_map = dict()
    for key, value in in_replay_buffer.data.items():
        chunks_map[key] = value.shape
        compressor_map[key] = lowdim_compressor

    print("Loading lowdim data")
    out_replay_buffer = ReplayBuffer.copy_from_store(
        src_store=in_replay_buffer.root.store,
        store=out_store,
        keys=lowdim_keys,
        chunks=chunks_map,
        compressors=compressor_map,
    )

    # worker function
    def put_img(zarr_arr, zarr_idx, img):
        try:
            zarr_arr[zarr_idx] = img
            # make sure we can successfully decode
            if verify_read:
                _ = zarr_arr[zarr_idx]
            return True
        except Exception:
            return False

    # TODO 接下来是图片处理工作
    n_cameras = 0
    camera_idxs = set()
    if image_keys is not None:
        n_cameras = len(image_keys)
        camera_idxs = set([0])  # * 在这里我们只有一个相机，这个参数其实没有用
    else:
        camera_idxs = set([0])
        n_cameras = 1

    n_steps = in_replay_buffer.n_steps
    episode_starts = (
        in_replay_buffer.episode_ends[:] - in_replay_buffer.episode_lengths[:]
    )
    episode_lengths = in_replay_buffer.episode_lengths
    # timestamps = in_replay_buffer['timestamp'][:]
    # dt = timestamps[1] - timestamps[0]
    with tqdm(
        total=n_steps * n_cameras, desc="Processing image data", mininterval=1.0
    ) as pbar:
        # 只处理一个相机的数据
        camera_idx = 0

        # 读取第一张图片来获取输入分辨率
        first_img = in_replay_buffer.data["3rd_bgr"][0]
        in_img_res = first_img.shape[:2]  # H W
        in_img_res = in_img_res[::-1]  # W H
        in_img_res = tuple(in_img_res)

        arr_name = f"camera_{camera_idx}"
        out_img_res = in_img_res
        if isinstance(out_resolutions, dict):
            if arr_name in out_resolutions:
                out_img_res = tuple(out_resolutions[arr_name])
        elif out_resolutions is not None:
            out_img_res = tuple(out_resolutions)

        # 创建输出数据集
        if arr_name not in out_replay_buffer:
            ow, oh = out_img_res
            _ = out_replay_buffer.data.require_dataset(
                name=arr_name,
                shape=(n_steps, oh, ow, 3),
                chunks=(1, oh, ow, 3),
                compressor=image_compressor,
                dtype=np.uint8,
            )
        arr = out_replay_buffer[arr_name]

        # 创建图像变换函数
        image_tf = get_image_transform(
            input_res=in_img_res, output_res=out_img_res, bgr_to_rgb=False
        )

        # 获取输入图像数据
        input_img_array = in_replay_buffer.data["3rd_bgr"][:]

        # 直接同步处理所有图像
        for step_idx in range(n_steps):
            # 获取原始图像
            input_img = input_img_array[step_idx]

            # 应用图像变换
            if in_img_res == out_img_res:
                # 如果分辨率相同，直接复制
                processed_img = input_img.copy()
            else:
                # 应用分辨率变换
                processed_img = image_tf(input_img)

            # 直接存储处理后的图像
            put_img(arr, step_idx, processed_img)

            # 更新进度
            pbar.update(1)

    print("Image processing completed!")
    return out_replay_buffer
