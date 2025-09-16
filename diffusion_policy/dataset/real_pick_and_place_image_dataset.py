import os
import sys

sys.path.append("/home/wxn/Projects/diffusion_policy")


import copy
import json
from logging import Logger
from typing import Dict

import cv2
import numpy as np
import torch
import zarr
from omegaconf import OmegaConf
from threadpoolctl import threadpool_limits

from diffusion_policy.common.normalize_util import get_image_range_normalizer
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.sampler import (
    SequenceSampler,
    downsample_mask,
    get_val_mask,
)
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import (
    LinearNormalizer,
    SingleFieldLinearNormalizer,
)
from diffusion_policy.real_world.real_data_conversion import real_data_to_replay_buffer
from diffusion_policy.real_world.my_real_data_conversion import my_real_data_to_replay_buffer

logger = Logger(__name__)


class RealPickAndPlaceImageDataset(BaseImageDataset):
    def __init__(
        self,
        shape_meta: dict,
        dataset_path: str,
        horizon=1,
        pad_before=0,
        pad_after=0,
        n_obs_steps=None,
        n_latency_steps=0,
        use_cache=False,
        seed=42,
        val_ratio=0.0,
        max_train_episodes=None,
        delta_action=False,
    ):
        assert os.path.isdir(dataset_path)

        replay_buffer = None
        shape_meta_json = json.dumps(OmegaConf.to_container(shape_meta), sort_keys=True)
        logger.info(f"Loading the ReplayBuffer from {dataset_path}")
        replay_buffer = _get_replay_buffer(
            dataset_path=dataset_path, shape_meta=shape_meta, store=zarr.MemoryStore()
        )

        if delta_action:
            actions = replay_buffer["action"][:]
            assert actions.shape[1] <= 3
            action_diff = np.zeros_like(actions)
            episode_ends = replay_buffer.episode_ends[:]
            for i in range(len(episode_ends)):
                start = 0
                if i > 0:
                    start = episode_ends[i - 1]
                end = episode_ends[i]
                action_diff[start + 1 : end] = np.diff(actions[start:end], axis=0)
            replay_buffer["action"][:] = action_diff

        rgb_keys = list()
        lowdim_keys = list()
        obs_shape_meta = shape_meta["obs"]
        for key, attr in obs_shape_meta.items():
            type = attr.get("type", "low_dim")
            if type == "rgb":
                rgb_keys.append(key)
            elif type == "low_dim":
                lowdim_keys.append(key)

        key_first_k = dict()
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )

        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask, max_n=max_train_episodes, seed=seed
        )

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon + n_latency_steps,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.val_mask = val_mask
        self.horizon = horizon
        self.n_latency_steps = n_latency_steps
        self.pad_before = pad_before
        self.pad_after = pad_after

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon + self.n_latency_steps,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=self.val_mask,
        )
        val_set.val_mask = ~self.val_mask
        return val_set

    def get_normalizer(self, **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # action
        normalizer["action"] = SingleFieldLinearNormalizer.create_fit(
            self.replay_buffer["action"]
        )

        # obs
        for key in self.lowdim_keys:
            normalizer[key] = SingleFieldLinearNormalizer.create_fit(
                self.replay_buffer[key]
            )

        # image
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self):
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        threadpool_limits(1)
        data = self.sampler.sample_sequence(idx)

        T_slice = slice(self.n_obs_steps)
        obs_dict = dict()
        for key in self.rgb_keys:
            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            del data[key]

        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        action = data["action"].astype(np.float32)

        # TODO 原本需要在这里用 n_latency_steps 跳过action延迟，但是由于我们的数据已经经过清洗，所以不需要这一步
        if self.n_latency_steps > 0:
            action = action[self.n_latency_steps :]

        torch_data = {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(action),
        }
        return torch_data


def zarr_resize_index_last_dim(zarr_arr, idxs):
    actions = zarr_arr[:]
    actions = actions[..., idxs]
    zarr_arr.resize(zarr_arr.shape[:-1] + (len(idxs),))
    zarr_arr[:] = actions
    return zarr_arr


def _get_replay_buffer(dataset_path, shape_meta, store):
    # 针对我们自己的数据格式处理replay buffer
    rgb_keys = list()
    lowdim_keys = list()
    out_resolutions = dict()
    lowdim_shapes = dict()
    obs_shape_meta = shape_meta["obs"]

    for key, attr in obs_shape_meta.items():
        type = attr.get("type", "low_dim")
        shape = tuple(attr.get("shape"))
        if type == "rgb":
            rgb_keys.append(key)
            c, h, w = shape  # C H W
            out_resolutions[key] = (w, h)
        elif type == "low_dim":
            lowdim_keys.append(key)
            lowdim_shapes[key] = tuple(shape)
            if "pose" in key:
                assert tuple(shape) in [(8,)]

    action_shape = tuple(shape_meta["action"]["shape"])
    assert action_shape in [(8,)]

    # load data
    cv2.setNumThreads(1)
    with threadpool_limits(1):
        replay_buffer = my_real_data_to_replay_buffer(
            dataset_path=dataset_path,
            out_store=store,
            out_resolutions=out_resolutions,
            lowdim_keys=lowdim_keys + ["action"],
            image_keys=rgb_keys,
        )

    if action_shape == (8,):
        zarr_arr = replay_buffer["action"]
        zarr_resize_index_last_dim(zarr_arr, idxs=[0, 1, 2, 3, 4, 5, 6, 7])

    for key, shape in lowdim_shapes.items():
        if "pose" in key and shape == (8,):
            zarr_arr = replay_buffer[key]
            zarr_resize_index_last_dim(zarr_arr, idxs=[0, 1, 2, 3, 4, 5, 6, 7])

    return replay_buffer


def test():
    import hydra
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    with hydra.initialize("../config"):
        cfg = hydra.compose("train_my_real_image_workspace.yaml")
        OmegaConf.resolve(cfg)
        dataset = hydra.utils.instantiate(cfg.task.dataset)

    from matplotlib import pyplot as plt

    normalizer = dataset.get_normalizer()
    nactions = normalizer["action"].normalize(dataset.replay_buffer["action"][:])
    diff = np.diff(nactions, axis=0)
    dists = np.linalg.norm(np.diff(nactions, axis=0), axis=-1)
    _ = plt.hist(dists, bins=100)
    plt.title("real action velocity")


if __name__ == "__main__":
    test()
