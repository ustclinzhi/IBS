import os 
import yaml
import genesis as gs
import numpy as np
import torch  
import argparse
import math
import inspect
import os
from os.path import join 
import re
import yaml 
import json  
import pickle
import sys
if sys.platform == "win32":
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath
from datetime import datetime 
from collections.abc import Mapping, Iterable
from dexmachina.asset_utils import get_rl_config_path 
from dexmachina.envs.base_env import BaseEnv
from dexmachina.envs.contacts import get_contact_marker_cfgs
from dexmachina.envs.constructors import get_common_argparser, parse_clip_string  
from dexmachina.rl.rl_games_wrapper import RlGamesVecEnvWrapper, RlGamesGpuEnv

from collections import defaultdict
import shutil
import moviepy
from rl_games.common import env_configurations, vecenv
from rl_games.common.algo_observer import IsaacAlgoObserver
from rl_games.torch_runner import Runner

def gather_object_state_tensor(demo_data):
    """ demo data should already be sliced since it's loaded from env kwargs """
    obj_pos = demo_data["obj_pos"]
    obj_quat = demo_data["obj_quat"]
    obj_arti = demo_data["obj_arti"] # need to reshape to (T, 1) instead of (T,)
    if len(obj_arti.shape) == 1:
        obj_arti = obj_arti[:, None]
    arr = np.concatenate([obj_pos, obj_quat, obj_arti], axis=1)
    return torch.tensor(arr).float()

def eval_one_episode(env, hand_motion_data, obj_state_tensor, print_rew=False, record_video=False, show_reference=False):
    obs = env.reset() 
    if isinstance(obs, dict):
        obs = obs["obs"]
    
    uenv = env.unwrapped
    ep_len = uenv.max_episode_length
    num_envs = uenv.num_envs
    device = torch.device('cuda:0')
    if record_video:
        uenv.start_recording() 
        uenv.max_video_frames = int(uenv.max_episode_length)
        max_frames = uenv.max_video_frames   
    
    obj = None
    if len(uenv.objects) > 0:
        obj = uenv.objects[uenv.object_names[0]]
        if obj.actuated:
            print("Setting eval time obj gains to 0.0")
            obj.set_joint_gains(0.0, 0.0, force_range=0.0)
    assert obj is not None, "No object found in the environment"
    
    left_hand = uenv.robots["left"]
    right_hand = uenv.robots["right"]
    
    # 获取手部关节名称
    left_joint_names = left_hand.joint_names
    right_joint_names = right_hand.joint_names
    
    eval_data = defaultdict(list)
    for i in range(ep_len):
        with torch.inference_mode():
            env_step = uenv.episode_length_buf.cpu().numpy()[0]
            
            # 使用预定义的手部运动系数而不是RL网络输出
            if env_step < len(hand_motion_data["left"]):
                left_joint_targets = hand_motion_data["left"][env_step]
                right_joint_targets = hand_motion_data["right"][env_step]
                
                # 设置左手关节位置
                left_hand.set_joint_position(
                    joint_targets=left_joint_targets[None, :],
                    env_idxs=[0],
                )
                
                # 设置右手关节位置
                right_hand.set_joint_position(
                    joint_targets=right_joint_targets[None, :],
                    env_idxs=[0],
                )
            
            # 执行仿真步骤（动作为空，因为我们直接设置了关节位置）
            actions = torch.zeros((num_envs, 0), device=device)  # 空动作
            obs, rew, dones, infos = env.step(actions) 
            
            # 添加必要的物理模拟步骤
            uenv.scene.step()

    frames = uenv.get_recorded_frames()
    return frames, {}


def main():
    parser = get_common_argparser()
    parser.add_argument('--checkpoint', '-ck', type=str, default="inspire_hand")
    parser.add_argument('--hand_motion_data', type=str, required=True, help="Path to hand motion data file")
    parser.add_argument('--output_render', '-or', action='store_true')
    parser.add_argument('--render_dir', '-out', type=str, default="rendered")
    parser.add_argument('--video_fname', '-of', type=str, default="-eval.mp4")
    
    args = parser.parse_args()

    ckpt_path = "/".join(args.checkpoint.split("/")[:-2])
    saved_cfg_fname = os.path.join(ckpt_path, "params", "env.pkl")
    
    ckpt_name = f"{args.checkpoint.split('/')[-1]}"
    run_name = os.path.basename(os.path.dirname(args.checkpoint))
    ckpt_data_folder = os.path.join(ckpt_path, ckpt_name.replace(".pth", "_mano_eval")).strip()
    os.makedirs(ckpt_data_folder, exist_ok=True)

    video_fname = join(ckpt_data_folder, f"video.mp4")
    if args.output_render:
        render_dir = os.path.join(args.render_dir, run_name)
        print('Saving video to a different folder')
        video_fname = os.path.join(render_dir, ckpt_name.split(".")[0] + args.video_fname)
        os.makedirs(render_dir, exist_ok=True)

    assert os.path.exists(saved_cfg_fname), f"File {saved_cfg_fname} does not exist"
    print(f"Loading saved env config from {saved_cfg_fname}")
    with open(saved_cfg_fname, "rb") as f:
        env_kwargs = pickle.load(f)
    print(env_kwargs['object_cfgs'])
    
    # 加载手部运动数据
    print(f"Loading hand motion data from {args.hand_motion_data}")
    hand_motion_data = np.load(args.hand_motion_data, allow_pickle=True).item()
    
    env_kwargs['robot_cfgs']['left']['urdf_path'] = r'E:\IBS\dexmachina\dexmachina\assets\inspire_hand\left_xyz_copy.urdf'
    env_kwargs['robot_cfgs']['right']['urdf_path'] = r'E:\IBS\dexmachina\dexmachina\assets\inspire_hand\right_xyz_copy.urdf'

    assert env_kwargs['env_cfg']['use_rl_games'], "The saved environment is not from rl-games"
    
    if args.raytrace and args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['raytrace'] = True

    env_kwargs['env_cfg']['early_reset_threshold'] = 0.0
    print("WARNING: setting env.is_eval to True")
    env_kwargs['env_cfg']['is_eval'] = True
    if args.vis:
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True
        env_kwargs['env_cfg']['scene_kwargs']['show_viewer'] = True
    print("Not applying external forces during eval time")
    env_kwargs['rand_cfg']['randomize'] = False 

    if args.show_markers:
        marker_cfgs = get_contact_marker_cfgs(
                num_vis_contacts=16,
                sources=['demo'],
                obj_parts=['top', 'bottom'],
                hand_sides=['left', 'right'],
            )
        env_kwargs['contact_marker_cfgs'] = marker_cfgs
        print('Setting visualize contact to True but observe contact force to False')
        env_kwargs['env_cfg']['scene_kwargs']['visualize_contact'] = True

    env_kwargs['env_cfg']['num_envs'] = args.num_envs
    env_kwargs['env_cfg']['rand_init_ratio'] = 0.0
    if args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True  
        env_kwargs['env_cfg']['record_video'] = True 
        print(f"Setting render resolution to 512") 
        env_kwargs['env_cfg']['camera_kwargs']['front'] = dict(
            res=(512, 512),
            fov=40,
            pos=(0.5, -1.5, 1.2),
            lookat=(0.0, -1.58, 2.0),
        ) 
    for name, cfg in env_kwargs['object_cfgs'].items():
        print("Setting eval time obj gains to 0.0")
        cfg['actuated'] = False

    print("demo_data :", env_kwargs['demo_data'].keys())
    newdemo_data = {
        'obj_pos': env_kwargs['demo_data']['obj_pos'],
        'obj_quat': env_kwargs['demo_data']['obj_quat'],
        'obj_arti': env_kwargs['demo_data']['obj_arti'],
        "contact_links_left": env_kwargs['demo_data']['contact_links_left'],
        "contact_links_right": env_kwargs['demo_data']['contact_links_right'],
        "contact_links_valid_left": env_kwargs['demo_data']['contact_links_valid_left'],
        "contact_links_valid_right": env_kwargs['demo_data']['contact_links_valid_right']
    }
    newdemo_data = recursively_to_tensor(newdemo_data)
    env_kwargs['demo_data'] = newdemo_data
    
    print("✅ demo_data 已转换为 Tensor。")
    
    # 删除重复赋值
    # env_kwargs['demo_data'] = newdemo_data
    if args.overlay:
        env_kwargs['env_cfg']['env_spacing'] = (0.0, 0.0)
    print("Removing any curriculum config during eval")
    # 确保安全地移除 curriculum_cfg，如果不存在也不会报错
    env_kwargs.pop("curriculum_cfg", None)
    device = torch.device('cuda:0')
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level='warning')

    env = BaseEnv(**env_kwargs)
    obj_state_tensor = gather_object_state_tensor(env_kwargs['demo_data'])

    # 转换手部运动数据为tensor格式
    hand_motion_tensor_data = {}
    for hand_side in ["left", "right"]:
        if hand_side in hand_motion_data:
            hand_motion_tensor_data[hand_side] = torch.tensor(hand_motion_data[hand_side], dtype=torch.float32)
        else:
            raise ValueError(f"Hand motion data for {hand_side} not found in the provided file")

    frames, eval_data = eval_one_episode(
        env, hand_motion_tensor_data, obj_state_tensor, False, args.record_video, False
    )
    if args.record_video: 
        from moviepy.editor import ImageSequenceClip
        clip = ImageSequenceClip(frames, fps=int(1/env.unwrapped.dt/2))
        clip.write_videofile(video_fname)
        print(f"Saved video to {video_fname}")
    
    print("Done evaluating")

def recursively_to_tensor(data):
    """递归地将列表/元组和字典中的数值数据转换为 torch.Tensor。"""
    
    if isinstance(data, torch.Tensor):
        return data # 已经是 Tensor
    
    if isinstance(data, np.ndarray):
        return torch.as_tensor(data)

    if isinstance(data, Mapping):
        return {key: recursively_to_tensor(value) for key, value in data.items()}
    
    if isinstance(data, (list, tuple)):
        try:
            return torch.as_tensor(data)
        except:
            new_data = [recursively_to_tensor(item) for item in data]
            return tuple(new_data) if isinstance(data, tuple) else new_data
    
    return data

if __name__ == '__main__':
    main()