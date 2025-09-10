from typing import Sequence, Tuple, Dict, Optional, Union
import os
import pathlib
import numpy as np
import av
import zarr
import numcodecs
import multiprocessing
import concurrent.futures
from tqdm import tqdm
from diffusion_policy.common.replay_buffer import ReplayBuffer, get_optimal_chunks
from diffusion_policy.common.cv2_util import get_image_transform
from diffusion_policy.real_world.video_recorder import read_video
from diffusion_policy.codecs.imagecodecs_numcodecs import (
    register_codecs,
    Jpeg2k
)
register_codecs()


def real_data_to_replay_buffer(
        dataset_path: str, 
        out_store: Optional[zarr.ABSStore]=None, 
        out_resolutions: Union[None, tuple, Dict[str,tuple]]=None, # 'camera_1': (width, height)
        lowdim_keys: Optional[Sequence[str]]=None, # [robot_eef_pose, action]
        image_keys: Optional[Sequence[str]]=None, # [camera_1, camera_3]
        lowdim_compressor: Optional[numcodecs.abc.Codec]=None,
        image_compressor: Optional[numcodecs.abc.Codec]=None,
        n_decoding_threads: int=multiprocessing.cpu_count(),
        n_encoding_threads: int=multiprocessing.cpu_count(),
        max_inflight_tasks: int=multiprocessing.cpu_count()*5,
        verify_read: bool=True
        ) -> ReplayBuffer: # 使用真实数据构建Replay Buffer
    """
    建议在调用此函数之前使用以下代码
    以避免CPU过度订阅
    cv2.setNumThreads(1)
    threadpoolctl.threadpool_limits(1)

    out_resolution:
        如果为 None:
            使用视频分辨率
        如果为 (width, height) 例如 (1280, 720)
        如果为字典:
            camera_0: (1280, 720)
    image_keys: ['camera_0', 'camera_1']
    """
    if out_store is None:
        out_store = zarr.MemoryStore()
    if n_decoding_threads <= 0:
        n_decoding_threads = multiprocessing.cpu_count()
    if n_encoding_threads <= 0:
        n_encoding_threads = multiprocessing.cpu_count()
    if image_compressor is None:
        image_compressor = Jpeg2k(level=50)

    # verify input
    input = pathlib.Path(os.path.expanduser(dataset_path))
    in_zarr_path = input.joinpath('replay_buffer.zarr') # 除了视频之外的数据
    in_video_dir = input.joinpath('videos')
    assert in_zarr_path.is_dir()
    assert in_video_dir.is_dir()
    # 构建replay buffer 包含 meta和data data中包含(action, robot_eef_pose, robot_eef_pose_vel, robot_joint, robot_joint_vel, state, timestamp)
    in_replay_buffer = ReplayBuffer.create_from_path(str(in_zarr_path.absolute()), mode='r') # ReplayBuffer实际上用的是self.root(zarr Group) 

    # save lowdim data to single chunk
    chunks_map = dict()
    compressor_map = dict()
    for key, value in in_replay_buffer.data.items():
        chunks_map[key] = value.shape # 保存所有数据的形状
        compressor_map[key] = lowdim_compressor

    print('Loading lowdim data')
    out_replay_buffer = ReplayBuffer.copy_from_store( # 只包含 robot_eef_pose 和 action 的replay buffer
        src_store=in_replay_buffer.root.store,
        store=out_store,
        keys=lowdim_keys,
        chunks=chunks_map,
        compressors=compressor_map
        )
    
    # worker function
    def put_img(zarr_arr, zarr_idx, img):
        try:
            zarr_arr[zarr_idx] = img
            # make sure we can successfully decode
            if verify_read:
                _ = zarr_arr[zarr_idx]
            return True
        except Exception as e:
            return False

    
    n_cameras = 0
    camera_idxs = set() 
    if image_keys is not None:
        n_cameras = len(image_keys)
        camera_idxs = set(int(x.split('_')[-1]) for x in image_keys) # 保留camera_后的数字作为编号
    else:
        # estimate number of cameras
        episode_video_dir = in_video_dir.joinpath(str(0))
        episode_video_paths = sorted(episode_video_dir.glob('*.mp4'), key=lambda x: int(x.stem))
        camera_idxs = set(int(x.stem) for x in episode_video_paths)
        n_cameras = len(episode_video_paths)
    
    n_steps = in_replay_buffer.n_steps # 一共数据集多少步 27672
    episode_starts = in_replay_buffer.episode_ends[:] - in_replay_buffer.episode_lengths[:] # 每一个样本的开始idx
    episode_lengths = in_replay_buffer.episode_lengths # 每一个样本的长度
    timestamps = in_replay_buffer['timestamp'][:]
    dt = timestamps[1] - timestamps[0]
    # 一共的图片数量 步骤数量 * 相机个数
    with tqdm(total=n_steps*n_cameras, desc="Loading image data", mininterval=1.0) as pbar:
        # one chunk per thread, therefore no synchronization needed
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_encoding_threads) as executor: #TODO 记得把线程回到 n_encoding_threads
            futures = set()
            for episode_idx, episode_length in enumerate(episode_lengths):
                episode_video_dir = in_video_dir.joinpath(str(episode_idx)) # 一个视频文件夹的地址
                episode_start = episode_starts[episode_idx]

                episode_video_paths = sorted(episode_video_dir.glob('*.mp4'), key=lambda x: int(x.stem))
                this_camera_idxs = set(int(x.stem) for x in episode_video_paths)
                if image_keys is None:
                    for i in this_camera_idxs - camera_idxs:
                        print(f"Unexpected camera {i} at episode {episode_idx}")
                for i in camera_idxs - this_camera_idxs:
                    print(f"Missing camera {i} at episode {episode_idx}")
                    if image_keys is not None:
                        raise RuntimeError(f"Missing camera {i} at episode {episode_idx}")

                for video_path in episode_video_paths:
                    camera_idx = int(video_path.stem)
                    if image_keys is not None:
                        # if image_keys provided, skip not used cameras
                        if camera_idx not in camera_idxs:
                            continue

                    # read resolution
                    with av.open(str(video_path.absolute())) as container:
                        video = container.streams.video[0]
                        vcc = video.codec_context
                        this_res = (vcc.width, vcc.height)
                    in_img_res = this_res

                    arr_name = f'camera_{camera_idx}'
                    # figure out save resolution
                    out_img_res = in_img_res
                    if isinstance(out_resolutions, dict):
                        if arr_name in out_resolutions:
                            out_img_res = tuple(out_resolutions[arr_name]) # camera_1 camera_3等
                    elif out_resolutions is not None:
                        out_img_res = tuple(out_resolutions)

                    # allocate array
                    if arr_name not in out_replay_buffer:
                        ow, oh = out_img_res
                        # 在out_replay_buffer.data中创建一个新数组
                        _ = out_replay_buffer.data.require_dataset(
                            name=arr_name,
                            shape=(n_steps,oh,ow,3),
                            chunks=(1,oh,ow,3),
                            compressor=image_compressor,
                            dtype=np.uint8
                        )
                    arr = out_replay_buffer[arr_name]

                    image_tf = get_image_transform(
                        input_res=in_img_res, output_res=out_img_res, bgr_to_rgb=False)
                    for step_idx, frame in enumerate(read_video(
                            video_path=str(video_path),
                            dt=dt,
                            img_transform=image_tf,
                            thread_type='FRAME',
                            thread_count=n_decoding_threads
                        )):
                        if len(futures) >= max_inflight_tasks:
                            # limit number of inflight tasks
                            completed, futures = concurrent.futures.wait(futures, 
                                return_when=concurrent.futures.FIRST_COMPLETED)
                            for f in completed:
                                if not f.result():
                                    raise RuntimeError('Failed to encode image!')
                            pbar.update(len(completed))
                        
                        global_idx = episode_start + step_idx
                        futures.add(executor.submit(put_img, arr, global_idx, frame))

                        if step_idx == (episode_length - 1):
                            break
            completed, futures = concurrent.futures.wait(futures)
            for f in completed:
                if not f.result():
                    raise RuntimeError('Failed to encode image!')
            pbar.update(len(completed))
    return out_replay_buffer

