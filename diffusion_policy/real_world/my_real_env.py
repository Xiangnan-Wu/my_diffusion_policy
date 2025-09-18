import pathlib

import numpy as np

from diffusion_policy.common.cv2_util import get_image_transform, optimal_row_cols


class MyRealEnv:
    def __init__(
        self,
        output_dir,
        FrankaArm,
        frequency=10,
        n_obs_steps=2,
        obs_image_resolution=(684, 272),
        img_capture_resolution=(684, 372),
        max_obs_buffer_size=30,
        obs_float32=False,
    ):
        output_dir = pathlib.Path(output_dir)
        color_tf = get_image_transform(
            input_res=img_capture_resolution,
            output_res=obs_image_resolution,
            bgr_to_rgb=False,  # 因为训练的时候就是bgr
        )
        color_transform = color_tf
        if obs_float32:
            color_transform = lambda x: color_tf(x).astype(np.float32) / 255

        def transform(data):
            data["color"] = color_transform(data["color"])
            return data

        rw, rh, col, row = optimal_row_cols(
            n_cameras=1,
            in_wh_ratio=obs_image_resolution[0] / obs_image_resolution[1],
            max_resolution=obs_image_resolution,
        )
        vis_color_transform = get_image_transform(
            input_res=img_capture_resolution, output_res=(rw, rh), bgr_to_rgb=False
        )
