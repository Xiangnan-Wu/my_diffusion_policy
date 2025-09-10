from typing import Optional, Callable, Generator
import numpy as np
import av
from diffusion_policy.common.timestamp_accumulator import get_accumulate_timestamp_idxs

def read_video(
        video_path: str,                                              # 视频文件路径
        dt: float,                                                    # 目标时间间隔(秒)，决定输出帧率
        video_start_time: float=0.0,                                  # 视频开始的绝对时间戳(秒)
        start_time: float=0.0,                                        # 数据采样开始时间(秒)
        img_transform: Optional[Callable[[np.ndarray], np.ndarray]]=None,  # 可选的图像变换函数
        thread_type: str="AUTO",                                      # 视频解码线程类型
        thread_count: int=0,                                          # 解码线程数量，0表示自动
        max_pad_frames: int=10                                        # 视频结束后用最后一帧填充的最大帧数
        ) -> Generator[np.ndarray, None, None]:
    """
    智能视频读取函数，按照指定时间间隔从视频中提取帧
    
    该函数的核心功能：
    1. 时间同步：将视频帧与目标时间网格对齐
    2. 采样控制：根据dt参数控制输出帧率
    3. 图像预处理：可选的图像变换（缩放、裁剪等）
    4. 序列填充：确保输出序列长度一致
    
    返回：
        Generator: 产生处理后的图像帧序列
    """
    
    frame = None  # 用于存储最后一帧，供后续填充使用
    
    # 打开视频文件
    with av.open(video_path) as container:
        # 获取第一个视频流
        stream = container.streams.video[0]
        
        # 配置解码参数以优化性能
        stream.thread_type = thread_type    # 设置线程类型：AUTO, FRAME, SLICE等
        stream.thread_count = thread_count  # 设置解码线程数，多线程可提升解码速度
        
        next_global_idx = 0  # 下一个全局索引，用于时间戳累积计算
        
        # 逐帧解码视频
        for frame_idx, frame in enumerate(container.decode(stream)):
            # 计算当前帧的绝对时间戳
            since_start = frame.time                      # 帧相对于视频开始的时间(秒)
            frame_time = video_start_time + since_start   # 帧的绝对时间戳
            
            # 时间戳累积和索引计算
            # 这是核心的时间同步机制
            local_idxs, global_idxs, next_global_idx = get_accumulate_timestamp_idxs(
                timestamps=[frame_time],        # 当前帧的时间戳（列表形式）
                start_time=start_time,          # 采样开始时间
                dt=dt,                          # 目标时间间隔
                next_global_idx=next_global_idx # 全局索引计数器
            )
            
            # 如果当前帧需要输出（时间戳匹配采样网格）
            if len(global_idxs) > 0:
                # 将视频帧转换为RGB格式的numpy数组
                array = frame.to_ndarray(format='rgb24')  # 形状: (H, W, 3)
                img = array
                
                # 如果提供了图像变换函数，则应用变换
                if img_transform is not None:
                    img = img_transform(array)  # 例如：缩放、裁剪、颜色转换等
                
                # 根据时间同步算法，可能需要重复输出同一帧
                # 这处理了视频帧率低于目标采样率的情况
                for global_idx in global_idxs:
                    yield img
    
    # 视频结束后的填充处理
    # 用最后一帧填充指定数量的帧，确保序列长度一致
    if frame is not None:  # 确保至少有一帧被处理
        array = frame.to_ndarray(format='rgb24')
        img = array
        if img_transform is not None:
            img = img_transform(array)
        
        # 重复输出最后一帧
        for i in range(max_pad_frames):
            yield img

class VideoRecorder:
    def __init__(self,
        fps,
        codec,
        input_pix_fmt,
        # options for codec
        **kwargs
    ):
        """
        input_pix_fmt: rgb24, bgr24 see https://github.com/PyAV-Org/PyAV/blob/bc4eedd5fc474e0f25b22102b2771fe5a42bb1c7/av/video/frame.pyx#L352
        """

        self.fps = fps
        self.codec = codec
        self.input_pix_fmt = input_pix_fmt
        self.kwargs = kwargs
        # runtime set
        self._reset_state()
    
    def _reset_state(self):
        self.container = None
        self.stream = None
        self.shape = None
        self.dtype = None
        self.start_time = None
        self.next_global_idx = 0
    
    @classmethod
    def create_h264(cls,
            fps,
            codec='h264',
            input_pix_fmt='rgb24',
            output_pix_fmt='yuv420p',
            crf=18,
            profile='high',
            **kwargs
        ):
        obj = cls(
            fps=fps,
            codec=codec,
            input_pix_fmt=input_pix_fmt,
            pix_fmt=output_pix_fmt,
            options={
                'crf': str(crf),
                'profile': profile
            },
            **kwargs
        )
        return obj


    def __del__(self):
        self.stop()

    def is_ready(self):
        return self.stream is not None

    def start(self, file_path, start_time=None):
        if self.is_ready():
            # if still recording, stop first and start anew.
            self.stop()

        self.container = av.open(file_path, mode='w')
        self.stream = self.container.add_stream(self.codec, rate=self.fps)
        codec_context = self.stream.codec_context
        for k, v in self.kwargs.items():
            setattr(codec_context, k, v)
        self.start_time = start_time
    
    def write_frame(self, img: np.ndarray, frame_time=None):
        if not self.is_ready():
            raise RuntimeError('Must run start() before writing!')
        
        n_repeats = 1
        if self.start_time is not None:
            local_idxs, global_idxs, self.next_global_idx \
                = get_accumulate_timestamp_idxs(
                # only one timestamp
                timestamps=[frame_time],
                start_time=self.start_time,
                dt=1/self.fps,
                next_global_idx=self.next_global_idx
            )
            # number of appearance means repeats
            n_repeats = len(local_idxs)
        
        if self.shape is None:
            self.shape = img.shape
            self.dtype = img.dtype
            h,w,c = img.shape
            self.stream.width = w
            self.stream.height = h
        assert img.shape == self.shape
        assert img.dtype == self.dtype

        frame = av.VideoFrame.from_ndarray(
            img, format=self.input_pix_fmt)
        for i in range(n_repeats):
            for packet in self.stream.encode(frame):
                self.container.mux(packet)

    def stop(self):
        if not self.is_ready():
            return

        # Flush stream
        for packet in self.stream.encode():
            self.container.mux(packet)

        # Close the file
        self.container.close()

        # reset runtime parameters
        self._reset_state()
