from typing import List, Optional, Union, Dict, Callable
import numbers
import time
import pathlib
from multiprocessing.managers import SharedMemoryManager
import numpy as np
import pyrealsense2 as rs
from collections import deque

from real_camera_utils import get_cam_extrinsic
from real_camera_utils import Camera

class ZedCam:
    def __init__(self, max_k=30):
        self.camera = Camera('3rd')
        self.frames = deque(maxlen=max_k)
        self.ts = deque(maxlen=max_k)
   
    
    @property
    def n_cameras(self):
        return 1

    @property
    def is_ready(self):
        is_ready = True
        return is_ready
    
    def get(self, k=None, out=None):
        this_frame = self.camera.capture()['3rd']['rgb'].copy()
        ts = time.time()
        self.frames.append(this_frame)
        self.ts.append(ts)
        if k is None or k > len(self.frames):
            k = len(self.frames)
        items = list(self.frames)[-k:]
        tss = list(self.ts)[-k:]
        if k == 0:
            return 
        