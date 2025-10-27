import time
from collections import deque


import dill
import hydra
import numpy as np
import rospy
import torch
from autolab_core import RigidTransform
from franka_interface_msgs.msg import SensorDataGroup
from frankapy import FrankaArm, SensorDataMessageType
from frankapy import FrankaConstants as FC
from frankapy.proto import PosePositionSensorMessage
from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg
from scipy.spatial.transform import Rotation as R




from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.real_world.real_camera_utils import Camera
from diffusion_policy.real_world.real_inference_util import (
  get_real_obs_dict,
  get_real_obs_resolution,
)
from diffusion_policy.workspace.base_workspace import BaseWorkspace




def publish_pose(pose:RigidTransform, id:int, timestamp:float, pub:rospy.Publisher, rate:rospy.Rate):
  """
  通过ros发布动作，实现连续操作，其中内涵了sleep
  """
  timestamp = timestamp
  traj_gen_proto_msg = PosePositionSensorMessage(
      id=id, timestamp=timestamp,
      position=pose.translation, quaternion=pose.quaternion
  )
  ros_msg = make_sensor_group_msg(
      trajectory_generator_sensor_msg=sensor_proto2ros_msg(
          traj_gen_proto_msg, SensorDataMessageType.POSE_POSITION),
  )
  # rospy.loginfo('Publishing: ID {}'.format(traj_gen_proto_msg.id))
  pub.publish(ros_msg)
  rate.sleep()


def get_gripper_state(franka, thres_hold):
   if franka.get_gripper_width() < thres_hold:
       return True #关闭
   else:
       return False


def main():
   # 超参数
   ckpt_path = "/media/casia/data4/wxn/diffusion_policy/my_diffusion_policy/checkpoints/lion_on_shelf.ckpt"
   frequency = 10
   step_per_inference = 6
  
  
   camera = Camera('3rd')
   franka = FrankaArm()
   frequency = 5
   dt = 1.0 / frequency
   duration = 90
   payload = torch.load(open(ckpt_path, 'rb'), pickle_module = dill)
   cfg = payload['cfg']
   cls = hydra.utils.get_class(cfg._target_)
   workspace = cls(cfg)
   workspace: BaseWorkspace
   workspace.load_payload(payload, exclude_keys = None, include_keys = None)
   franka.reset_joints()
   rate = rospy.Rate(frequency)
   publisher = rospy.Publisher(FC.DEFAULT_SENSOR_PUBLISHER_TOPIC,
                               SensorDataGroup,
                               queue_size = 20)
   gripper_state = False # False 夹爪打开 True 夹爪闭合


   #* 初始化夹爪状态
   rospy.loginfo(" 初始化夹爪状态 ")
   franka.open_gripper()
   rospy.sleep(1.0)
   rospy.loginfo(" 夹爪初始化完成 ")
  
   # 训练时只采用了 train_diffusion_unet_image
   action_offset = 0
   delta_action = False
   if "diffusion" in cfg.name:
       # diffusion model
       policy: BaseImagePolicy
       policy = workspace.model
       if cfg.training.use_ema:
           policy = workspace.ema_model
          
       device = torch.device("cuda")
       policy.eval().to(device)
      
       # 设置推理参数
       policy.num_inference_steps = 16
       policy.n_action_steps = policy.horizon - policy.n_obs_steps + 1
      
   dt = 1/frequency
   obs_res = get_real_obs_resolution(cfg.task.shape_meta)
   n_obs_steps = cfg.n_obs_steps
   print("n_obs_steps: ", n_obs_steps)
   print("steps_per_inference:", step_per_inference)
   print("action_offset:", action_offset)
  
   # ========= DP control ============
   policy.reset()
   print("开始Diffusion Policy 策略控制")
   iter_idx = 0
   term_area_start_timestamp = float('inf')
   prev_target_pose = None
   action_id = 0
   obs_buffer = deque(maxlen = 2)
  
   curr_pose = franka.get_pose()
   franka.goto_pose(curr_pose,
                        duration = duration,
                        dynamic = True,
                        buffer_time = 10,
                        cartesian_impedances = [600.0, 600.0, 600.0, 50.0, 50.0, 50.0])
   while True:
       print("获取环境观测：相机图像 + 机械臂自身状态")
       # 相机拍照
       result_dict = camera.capture()
       # 机械臂自身状态
       robot_eef_pose = franka.get_pose()
       translation = robot_eef_pose.translation
       quaternion = robot_eef_pose.quaternion #! w x y z
       # Gripper state True -> closed, False -> opened
       gripper_state = get_gripper_state(franka, 0.075)
       proprio = np.concatenate([translation, quaternion, np.array([float(gripper_state)])])
      
       # 构建obs
       current_obs = {
           'camera_0': result_dict['3rd']['rgb'],
           'robot_eef_pose': proprio
       }
       #* 历史观测长度为2
       obs_buffer.append(current_obs)
       if len(obs_buffer) < 2:
           continue
      
       env_obs = {
           'camera_0': np.stack([obs_buffer[0]['camera_0'],
                                 obs_buffer[1]['camera_0']], axis = 0),
           'robot_eef_pose': np.stack([obs_buffer[0]['robot_eef_pose'],
                                       obs_buffer[1]['robot_eef_pose']], axis = 0),
       }
      
       obs_dict_np = get_real_obs_dict(env_obs, cfg.task.shape_meta)
       #! 形状能匹配吗
       obs_dict = dict_apply(obs_dict_np, lambda x: torch.from_numpy(x).unsqueeze(0).to(device))
       result = policy.predict_action(obs_dict)
       action = result['action'][0].detach().to('cpu').numpy()
       #! 不需要delta aciton
       target_pose = franka.get_pose()
      
       for i in range(action.shape[0]):
           target_pose = action[i]
           rotation_matrix = R.from_quat([target_pose[4], target_pose[5], target_pose[6],target_pose[3]]).as_matrix()
           target_pose = RigidTransform(rotation=rotation_matrix, translation=target_pose[:3])
           # target_pose = RigidTransform(translation = target_pose[:3], quaternion = target_pose[3:7])
           timestamp = time.time()
           publish_pose(target_pose, action_id, timestamp, pub=publisher, rate = rate)
           target_gripper_state = float(action[i][-1])
           if target_gripper_state > 0.75 and not gripper_state:
               franka.close_gripper()
               time.sleep(1.0)
               gripper_state = True
           elif target_gripper_state < 0.75 and gripper_state:
               franka.open_gripper()
               time.sleep(1.0)
               gripper_state = False
           action_id += 1
           rate.sleep()
      


if __name__ == '__main__':
   main()

