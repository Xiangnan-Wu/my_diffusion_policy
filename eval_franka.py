import time
from collections import deque
from typing import Any

import click
import cv2
import dill
import hydra
import numpy as np
import scipy.spatial.transform.rotation as R
import torch
from autolab_core import RigidTransform
from frankapy import FrankaArm
from omegaconf import OmegaConf

from diffusion_policy.common.precise_sleep import precise_wait
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.real_world.real_camera_utils import Camera
from diffusion_policy.real_world.real_inference_util import get_real_obs_dict
from diffusion_policy.workspace.base_workspace import BaseWorkspace

OmegaConf.register_new_resolver("eval", eval, replace=True)


# ========== 需要你实现的3个硬件接口函数 ==========
def init_camera():
    """返回你的相机句柄，用于 capture_frame().
    例如返回你在 eval_reference.py 中的 Camera 实例。
    """
    camera = Camera("3rd")
    return camera


def capture_frame(camera_handle):
    """从相机抓取一帧，返回 img(H,W,3) numpy.
    - 通道顺序需与训练一致（若训练BGR就返回BGR）。
    """
    result_dict = camera_handle.capture()
    return result_dict["3rd"]["rgb"]


def get_gripper_state(robot_handle, thres_hold):
    width = robot_handle.get_gripper_width()
    if width > thres_hold:
        return False
    else:
        return True


def get_robot_pose8(robot_handle):
    """读取当前末端位姿，返回 [x,y,z, rx,ry,rz] 轴角(弧度)。"""
    pose = robot_handle.get_pose()
    gripper = get_gripper_state(robot_handle, 0.075)
    pose8 = np.concatenate([pose.translation, pose.quaternion, [float(gripper)]])
    return pose8


def goto_pose8(robot_handle, pose8: np.ndarray, duration: float):
    """发送6维目标位姿到机械臂，持续时间 duration 秒（可近似）。"""
    rigid_transform = RigidTransform(
        translation=pose8[:3],
        rotation=R.from_quat(np.array([pose8[4], pose8[5], pose8[6], pose8[3]])),
    )
    robot_handle.goto_pose(rigid_transform, duration=duration)
    gripper_state = get_gripper_state(robot_handle, 0.075)
    if gripper_state != pose8[-1] and pose8[-1] == 0.0:
        robot_handle.open_gripper()
    elif gripper_state != pose8[-1] and pose8[-1] == 1.0:
        robot_handle.close_gripper()
    time.sleep(1)  # 等待夹爪


def init_robot():
    """返回你的 FrankaArm 句柄（或控制封装）。"""
    robot = FrankaArm()
    return robot


# ========== 你需要根据动作维度设计的映射 ==========
def map_action_seq_to_pose_seq(
    action_seq: np.ndarray, curr_pose6: np.ndarray
) -> np.ndarray:
    """将策略输出的 action 序列映射为 6DoF 目标位姿序列。
    - action_seq: (T, Da)
    - curr_pose6: (6,)
    返回: target_poses (T, 6)
    提示：你的模型动作维度为8（见打印），请按你的训练定义实现此函数。
    例如使用 Δxyz+Δrpy 累加，或直接绝对位姿等。
    """
    raise NotImplementedError("请根据你的动作定义实现 map_action_seq_to_pose_seq()")


@click.command()
@click.option("--input", "-i", required=True, help="Path to checkpoint")
@click.option("--output", "-o", required=True, help="Directory to save recording")
@click.option(
    "--init_joints",
    "-j",
    is_flag=True,
    default=False,
    help="Whether to initialize robot joint configuration in the beginning.",
)
@click.option(
    "--steps_per_inference",
    "-si",
    default=6,
    type=int,
    help="Action horizon for inference.",
)
@click.option(
    "--max_duration", "-md", default=60, help="Max duration for each epoch in seconds."
)
@click.option(
    "--frequency", "-f", default=10, type=float, help="Control frequency in Hz."
)
def main(input, output, init_joints, steps_per_inference, max_duration, frequency):
    ckpt_path = input
    payload = torch.load(open(ckpt_path, "rb"), pickle_module=dill)
    cfg = payload["cfg"]
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    action_offset = 0
    if "diffusion" in cfg.name:
        policy: BaseImagePolicy
        workspace_any: Any = workspace
        policy = workspace_any.model
        if cfg.training.use_ema:
            policy = workspace_any.ema_model

        device = torch.device("cuda")
        policy.eval().to(device)

        # 设置推理参数
        policy_any: Any = policy
        setattr(policy_any, "num_inference_steps", 16)
        setattr(
            policy_any,
            "n_action_steps",
            policy_any.horizon - policy_any.n_obs_steps + 1,
        )

        # 设置时序参数
        dt = 1 / frequency
        n_obs_steps = cfg.n_obs_steps
        print("n_obs_steps: ", n_obs_steps)
        print("steps_per_inference:", steps_per_inference)
        print("action_offset:", action_offset)

        cv2.setNumThreads(1)

        # ========== 第1步：初始化相机与机器人（你来实现上述函数） ==========
        camera = init_camera()
        robot = init_robot()

        # ========== 第2步：准备帧缓存，收集 n_obs_steps 帧 ==========
        img_buf = deque(maxlen=n_obs_steps)
        ts_buf = deque(maxlen=n_obs_steps)

        def push_one_frame():
            img = capture_frame(camera)
            ts = time.time()
            img_buf.append(img)
            ts_buf.append(ts)

        while len(img_buf) < n_obs_steps:
            push_one_frame()
            time.sleep(max(0.0, dt))

        # ========== 第3步：预热一次前向 ==========
        curr_pose8 = get_robot_pose8(robot)
        env_obs = {
            "camera_0": np.stack(list(img_buf), axis=0),  # (T,H,W,3)
            "robot_eef_pose": np.tile(curr_pose8[None, :], (n_obs_steps, 1)),
            "timestamp": np.asarray(list(ts_buf), dtype=np.float64),
        }
        obs_dict_np = get_real_obs_dict(env_obs=env_obs, shape_meta=cfg.task.shape_meta)
        obs_dict = dict_apply(
            obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
        )  # type: ignore[arg-type]
        with torch.no_grad():
            _ = policy.predict_action(obs_dict)

        print("Ready! Start control loop...")

        # ========== 第4步：主循环：取帧→组装obs→推理→动作映射→下发 ==========
        t_start = time.monotonic()
        iter_idx = 0
        try:
            while True:
                # 时间控制（按块提交 steps_per_inference 步）
                # 抓取新帧
                push_one_frame()
                curr_pose8 = get_robot_pose8(robot)

                # 组装 obs 并推理
                env_obs = {
                    "camera_0": np.stack(list(img_buf), axis=0),
                    "robot_eef_pose": np.tile(curr_pose8[None, :], (n_obs_steps, 1)),
                    "timestamp": np.asarray(list(ts_buf), dtype=np.float64),
                }
                obs_dict_np = get_real_obs_dict(
                    env_obs=env_obs, shape_meta=cfg.task.shape_meta
                )
                obs_dict = dict_apply(
                    obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device)
                )  # type: ignore[arg-type]

                with torch.no_grad():
                    s = time.time()
                    result = policy.predict_action(obs_dict)
                    action_seq = (
                        result["action"][0].detach().to("cpu").numpy()
                    )  # (L, Da)
                    inf_latency = time.time() - s
                print(f"Inference latency: {inf_latency:.3f}s, steps={len(action_seq)}")

                # 下发执行：假设动作为8维绝对位姿+夹爪，直接调用 goto_pose8
                for i in range(len(action_seq)):
                    goto_pose8(robot, action_seq[i], duration=max(0.02, dt))
                    precise_wait(time.monotonic() + dt)

                iter_idx += steps_per_inference

                # 可选：超时退出
                if (time.monotonic() - t_start) > max_duration:
                    print("Terminated by timeout.")
                    break

        except KeyboardInterrupt:
            print("Interrupted!")
        finally:
            print("Stopped.")


if __name__ == "__main__":
    main(
        input="/home/wxn/Projects/diffusion_policy/data/outputs/2025.09.17/02.41.38_train_diffusion_unet_image_close_upper_drawer/checkpoints/latest.ckpt",
        output="/home/wxn/Projects/diffusion_policy/data/eval_output",
        init_joints=False,
        steps_per_inference=6,
        max_duration=60,
        frequency=10,
    )
