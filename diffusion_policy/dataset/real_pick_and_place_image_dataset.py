from typing import Dict
import torch
import numpy as np
import copy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from diffusion_policy.model.common.normalizer import LinearNormalizer
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.common.normalize_util import get_image_range_normalizer

class RealPickAndPlaceImageDataset(BaseImageDataset):
    def __init__(self, 
                 data_path,
                 horizon = 1,
                 pad_before= 0
                 pad_after = 0,
                 seed = 42,
                 val_ratio = 0.0,
                 max_train_episodes = None):
        super().__init__()
        