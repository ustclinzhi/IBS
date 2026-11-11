"""
Load and inspect the raw URDF files into a minimal Genesis scene

- Iterate all the rotations to find proper camera 
URDF=filename
python hand_proc/inspect_raw_urdf.py --record_video --render_image --urdf_path $URDF --skip_wrist_interp  --group_collisions --interp_step 30 --iterate_quat
python hand_proc/inspect_raw_urdf_copy.py --urdf_path assets/mano_double/rhand_mano_low_mass.urdf -v --interp_steps 500
"""
import sys
if sys.platform == "win32":
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath
import os
import cv2 
import torch
import argparse
import numpy as np
import genesis as gs 
from glob import glob 
from os.path import join 
from scipy.spatial.transform import Rotation
from dexmachina.hand_proc.hand_utils import generate_90degree_rotation_quaternions
from dexmachina.envs.object import ArticulatedObject
import pickle
from dexmachina.envs.base_env_copy import BaseEnv
from collections.abc import Mapping, Iterable
import trimesh
pi=3.1415926
WRISTTRANS_OFFSET_LEFT=np.array([0.0956699401140213, -0.006383428350090981, -0.006186304613947868])-np.array([-0.01,0,-0.028])
WRISTTRANS_OFFSET_RIGHT=np.array([-0.0956699401140213, -0.006383428350090981, -0.006186304613947868])-np.array([0,0,-0.005])
URDFSCALE=1.0
def correct_quaternion(q):
    """把mano手参数45维里的关节旋转参数限制到[0,pi内]"""
    for i in range(15):
        qx = q[3*i]
        qy = q[3*i+1]
        qz = q[3*i+2]
        normq=[qx, qy, qz]
        normq = np.array(normq)
        qrot=np.linalg.norm(normq)
        if qrot>2*pi:
            qrotnew = qrot - 2*pi
            qsacle = qrotnew / qrot
            q[3*i] = normq[0] * qsacle
            q[3*i+1] = normq[1] * qsacle
            q[3*i+2] = normq[2] * qsacle
        elif qrot>pi:
            qrotnew = 2*pi - qrot
            qsacle = -qrotnew / qrot
            q[3*i] = normq[0] * qsacle
            q[3*i+1] = normq[1] * qsacle
            q[3*i+2] = normq[2] * qsacle


def recursively_to_tensor(data):
    """递归地将列表/元组和字典中的数值数据转换为 torch.Tensor。"""
    
    if isinstance(data, torch.Tensor):
        return data # 已经是 Tensor
    
    if isinstance(data, np.ndarray):
        # 从 NumPy 数组创建 Tensor
        return torch.as_tensor(data)

    if isinstance(data, Mapping):
        # 字典：递归处理所有值
        return {key: recursively_to_tensor(value) for key, value in data.items()}
    
    if isinstance(data, (list, tuple)):
        # 列表/元组：尝试直接创建 Tensor
        try:
            # 这是一个关键步骤，它解决了列表没有 .shape 的问题
            return torch.as_tensor(data)
        except:
            # 如果失败（例如列表元素类型不一致），则递归处理元素
            new_data = [recursively_to_tensor(item) for item in data]
            # 保持原结构类型（列表或元组）
            return tuple(new_data) if isinstance(data, tuple) else new_data
    
    # 其他类型（str, int, float, bool 等），直接返回
    return data
def render_image(camera, transparent=True):
    """Render an image from the camera"""
    img, depth_arr, seg_arr, normal_arr = camera.render(segmentation=True)
    rgb_img = img.copy()
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)  # only do this for vis!
    if transparent: # return 4-channel
        max_id = seg_arr.max() # assuming this is background 
        channel = np.ones(img.shape[0:2], dtype=np.uint8) * 255
        channel[seg_arr == max_id] = 0
        channel[seg_arr == -1] = 0
        img = np.concatenate([img, channel[:, :, None]], axis=-1)
        # full the background in rgb_img with 255
        rgb_img[seg_arr == max_id] = 255
        rgb_img[seg_arr == -1] = 255
    return img, rgb_img

def get_base_rotation(name):
    """ ASSUME ALL LEFT HANDS"""
    if 'ability' in name and 'left' in name:
        return (0.7071, 0, 0, -0.7071)
    if 'allegro' in name and 'left' in name:
        return (1, 0, 0, 0)
    if 'dex' in name and 'left' in name:
        return (1, 0, 0, 0)
    if 'inspire' in name and 'left' in name:
        return (0, 0, 0.7071, -0.7071)
    if 'mimic' in name and 'left' in name:
        return (0.5, 0.5, 0.5, 0.5)
    if 'schunk' in name and 'left' in name:
        return (0.7071, 0, 0, -0.7071)
    if 'xhand' in name and 'left' in name:
        return (0, 0.7071, -0.7071, 0)
    if 'shadow' in name and 'left' in name:
        return (1, 0, 0, 0)


def interpolate_hand_joints(hand, n_steps=200, skip_wrist=False):
    all_joints = hand.joints
    actuated_joints = [joint for joint in all_joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]] 
    act_idxs = [joint.dof_idx_local for joint in actuated_joints]
    for joint in actuated_joints:
        print(f"Joint {joint.name} dof: {joint.dof_idx_local} ")
    limits = []
    for joint in actuated_joints:
        if joint.type in [gs.JOINT_TYPE.REVOLUTE]:
            limits.append(joint.dofs_limit[0])
        else:
            limits.append((-0.05, 0.05))
    # overwrite rot limits
    # limits[3:6] = [
    #     (-1.8, 0),
    #     # (0.0, 0.1),
    #     # (3.0, 3.2),
    #     (0.68, 5),
    #     # (3.6, 3.7),
    #     # (1.68, 1.8),
    #     (0, 4.14)
    # ]
    lower_bounds = [limit[0] for limit in limits]
    upper_bounds = [limit[1] for limit in limits]
    if skip_wrist:
        # still actuate it but just set limit to 0
        lower_bounds[:6] = [0] * 6
        upper_bounds[:6] = [0] * 6
    # interpolate between the lower and upper bounds 
    interp_joints = [] 
    for idx, (lower, upper) in enumerate(zip(lower_bounds, upper_bounds)):
        values1 = np.linspace(lower, upper, n_steps) 
        # then iterate from upper to lower again  
        values2 = np.linspace(upper, lower, n_steps)
        interp_joints.append(
            np.concatenate([values1, values2])
        )
    interp_joints = np.stack(interp_joints, axis=1) # shape (n_steps, n_joints)
    return interp_joints, act_idxs

def interpolate_wrist_finger_separately(hand, n_steps=50, skip_wrist=False, wrist_only=False, wrist_rot_only=False):
    """ 
    Interpolate only one joint at a time for the first 6 wrist joints, then the finger joints
    returns 2 * n_steps * 7 interpolated joint values
    """
    all_joints = hand.joints
    actuated_joints = [joint for joint in all_joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]
    if skip_wrist:
        actuated_joints = actuated_joints[6:]   
    if wrist_only:
        actuated_joints = actuated_joints[:6]
    if wrist_rot_only:
        actuated_joints = actuated_joints[3:6]
    act_idxs = [joint.dof_idx_local for joint in actuated_joints]
    limits = []
    for joint in actuated_joints:
        if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]:
            limits.append(joint.dofs_limit[0])
        # else:
        #     limits.append((-0.08, 0.08))
    if wrist_rot_only: 
        # mimic left hand
        # limits = [
        #     (-0.1, 0.1), 
        #     (2.8, 2.9), 
        #     (1.2, 1.3),
        # ]
        # mimic right:
        limits = [
            (-0.3, -0.2),
            (1.8, 1.9), 
            (0, 0.9),
        ]
    lower_bounds = [limit[0] for limit in limits]
    upper_bounds = [limit[1] for limit in limits]
    # default_vals = [(lower + upper) / 2 for lower, upper in zip(lower_bounds, upper_bounds)]
    # default_vals = np.array(default_vals)
    num_joints = len(actuated_joints)
    # interpolate between the lower and upper bounds 
    interp_joints = []
    for i in range(len(lower_bounds)):
        lower, upper = lower_bounds[i], upper_bounds[i]
        print(lower, upper)
        values1 = np.linspace(lower, upper, n_steps) 
        # then iterate from upper to lower again  
        values2 = np.linspace(upper, lower, n_steps)
        joints = np.zeros((n_steps * 2, num_joints))  
        joints[:, i] = np.concatenate([values1, values2])
        # fill other joints with the lower bound limit value:
        for j in range(num_joints):
            if j != i:
                joints[:, j] = lower_bounds[j]
        interp_joints.append(joints)
    
    # now interpolate the finger joints
    joints = np.zeros((n_steps * 2, num_joints))

    for i in range(6, num_joints):
        lower, upper = lower_bounds[i], upper_bounds[i]
        values1 = np.linspace(lower, upper, n_steps) 
        # then iterate from upper to lower again  
        values2 = np.linspace(upper, lower, n_steps)
        joints[:, i] = np.concatenate([values1, values2])
        
    interp_joints.append(joints) 
    interp_joints = np.concatenate(interp_joints, axis=0) # shape (n_steps, n_joints)
    print(act_idxs)
    return interp_joints, act_idxs

def export_video(frames, path, fps=15):
    if len(frames) == 0:
        return 
    path = path + ".mp4" if not path.endswith(".mp4") else path
    from moviepy.editor import ImageSequenceClip
    clip = ImageSequenceClip(frames, fps=fps)
    clip.write_videofile(path)

def gather_geom_link_groups(hand):
    # for each geom on the hand entity, get its link's global and local indx and name
    info = dict()
    for geom in hand.geoms:
        geom_id = geom.idx
        link = geom.link
        link_local_idx = link.idx_local
        link_name = link.name
        info[geom_id] = (link_local_idx, link_name)
    return info

def find_all_parents(link, all_links):
    """recursively find all the parents global_idx of a link"""
    parents = [] 
    # print('current link:', link.name, link.idx, link.parent_idx)
    # if link.name != 'base_link':
    #     breakpoint()
    if link.idx == -1 or link.parent_idx == -1: # root link for the entity has -1 parent
        return parents
    else:
        return find_all_parents(all_links[link.parent_idx], all_links) + [link.parent_idx]

def divide_hand_groups(hand, palm_link_name='base_link'):
    """
    for each hand, we have 2 kinds of collision groups for each geom
    1. palm link group: the geom's link local idx <= the palm link idx (assume the wrist & palm are the first few links)
    2. finger groups: each group has one link whose parent is the palm link (this should use the global idx for palm)
    """
    palm_links = [link for link in hand.links if link.name == palm_link_name]
    assert len(palm_links) == 1, f"Found {len(palm_links)} palm links"
    palm_idx_local = palm_links[0].idx_local
    palm_idx_global = palm_links[0].idx
    finger_groups = {geom.link.idx_local: 0 for geom in hand.geoms if geom.link.idx_local <= palm_idx_local} # {geom's link local idx: group_id}
    
    first_links = []
    for link in hand.links:
        if link.parent_idx == palm_idx_global:
            group_id = len(first_links) + 1
            first_links.append(
                (link.idx, link.idx_local, link.name, group_id)
            )  
    print("Found first links:", first_links)
    all_links = hand.solver.links # this should be all global links!! stuck in infinite recursion if use only entity links
    for geom in hand.geoms: 
        link = geom.link
        parent_global_idxs = find_all_parents(link, all_links)
        for first_link in first_links:
            idx, idx_local, name, group_id = first_link
            if idx in parent_global_idxs:
                finger_groups[geom.link.idx_local] = group_id
                break
    return finger_groups

def main(args):
    with open(r"E:\IBS\dexmachina\arctic\data\arctic_data\data\raw_seqs\s01\box_use_01.mano.npy", "rb") as f:
        hand_mano_data=np.load(f, allow_pickle=True).item()
    maxstep=hand_mano_data["left"]["rot"].shape[0]
    for i in range(maxstep):
        correct_quaternion(hand_mano_data["right"]["pose"][i])
        correct_quaternion(hand_mano_data["left"]["pose"][i])
        print(hand_mano_data["right"]["pose"][i])
    print(hand_mano_data["right"]["pose"][131])
    # hand_mano_data["right"]["pose"][0]=[0]*45
    # hand_mano_data["right"]["trans"][0]=[0]*3
    # hand_mano_data["right"]["rot"][0]=[0]*3
    # hand_mano_data["left"]["pose"][0]= [0]*45
    # hand_mano_data["left"]["trans"][0]=[0]*3 
    # hand_mano_data["left"]["rot"][0]=[0]*3

 

    hands_list = args.hands_list
    hand_names = []
    for hand in hands_list:
        if '_hand' in hand or 'xhand' == hand:
            hand_names.append(hand)
        else:
            hand_names.append(hand + '_hand')
    num_hands = len(hand_names)
    urdfs = dict()
    if len(args.urdf_path) > 0:
        assert os.path.exists(args.urdf_path), f"URDF path {args.urdf_path} does not exist"
        hand = args.urdf_path.split('assets/')[-1].split('/')[0]
        hand += '_left' if 'left' in args.urdf_path else '_right'
        urdfs[hand] = args.urdf_path
        hand_names = [hand]
    else:
        for hand in hand_names:
            urdf_paths = glob(join(args.asset_path, hand, "*.urdf")) + glob(join(args.asset_path, hand, "*", "*.urdf")) + glob(join(args.asset_path, hand, "*", "*", "*.urdf"))
            if 'barrett' not in hand:
                urdf_paths = [path for path in urdf_paths if args.side in path]
            # if args.vis_6dof:
            #     # filter only fnames that contain 6dof wrist joints
            #     urdf_paths = [path for path in urdf_paths if '6dof' in path]
            # else:
            #     urdf_paths = [path for path in urdf_paths if '6dof' not in path]

            # urdf_paths = glob(join(args.asset_path, hand,  "*copy.urdf")) 

            assert len(urdf_paths) >= 1, f"Found {len(urdf_paths)} URDFs for {hand} - {args.side}"
            _path = urdf_paths[0]
            urdfs[hand] = _path






    saved_cfg_fname = "E:\IBS\dexmachina\dexmachinatest12.22\inspire_hand\inspire-test_box30-230-s01-u01_B12000_hybrid_thres0.5_ho16_imi0.2_con2.0_bc0.2_ibsbeta25.0\params\env.pkl"

    with open(saved_cfg_fname, "rb") as f:
        env_kwargs = pickle.load(f)
    print(env_kwargs['object_cfgs'])
    env_kwargs['object_cfgs'] =  {'box': {'name': 'box', 'base_init_pos': [0.0597, -0.2476, 1.0354], 'base_init_quat': [-0.6413, 0.2875, 0.6467, -0.2964], 'base_init_qpos': [0.0], 'num_sample_vertics': 300, 'convexify': False, 'fixed': False, 'actuated': True, 'kp': 80.0, 'kv': 5.0, 'force_range': 50.0, 'collect_data': False, 'offset_pos': [0.0, 0.0, 0.0], 'color': None, 'show_link_frame': False, 'urdf_path': r'E:\IBS\dexmachina\dexmachina\assets\arctic\box\decomp\box_decomp.urdf', 'bottom_mesh_fname': r'E:\IBS\dexmachina\dexmachina\assets\arctic\box\bottom_watertight_tiny.stl', 'top_mesh_fname': r'E:\IBS\dexmachina\dexmachina\assets\arctic\box\top_watertight_tiny.stl'}}



 
    #{'box': {'name': 'box', 'base_init_pos': [0.0597, -0.2476, 1.0354], 'base_init_quat': [-0.6413, 0.2875, 0.6467, -0.2964], 'base_init_qpos': [0.0], 'num_sample_vertics': 300, 'convexify': False, 'fixed': False, 'actuated': True, 'kp': 80.0, 'kv': 5.0, 'force_range': 50.0, 'collect_data': False, 'offset_pos': [0.0, 0.0, 0.0], 'color': None, 'show_link_frame': False, 'urdf_path': r'E:\IBS\dexmachina\dexmachina\assets\arctic\box\decomp\box_decomp.urdf', 'bottom_mesh_fname': r'E:\IBS\dexmachina\dexmachina\assets\arctic\box\bottom_watertight_tiny.stl', 'top_mesh_fname': r'E:\IBS\dexmachina\dexmachina\assets\arctic\box\top_watertight_tiny.stl'}}

    #{'waffleiron': {'name': 'waffleiron', 'base_init_pos': [0.0, 0.0, 0.3], 'base_init_quat': [1.0, 0.0, 0.0, 0.0], 'base_init_qpos': [0.0], 'num_sample_vertics': 300, 'convexify': False, 'fixed': False, 'actuated': True, 'kp': 80.0, 'kv': 5.0, 'force_range': 50.0, 'collect_data': False, 'offset_pos': [0.0, 0.0, 0.0], 'color': None, 'show_link_frame': False, 'urdf_path': r'E:\IBS\dexmachina\dexmachina\assets\arctic\waffleiron\decomp\waffleiron_decomp.urdf', 'bottom_mesh_fname': r'E:\IBS\dexmachina\dexmachina\assets\arctic\waffleiron\bottom_watertight_tiny.stl', 'top_mesh_fname': r'E:\IBS\dexmachina\dexmachina\assets\arctic\waffleiron\top_watertight_tiny.stl'}}
    #{'mixer': {'name': 'mixer', 'base_init_pos': [0.0, 0.0, 0.3], 'base_init_quat': [1.0, 0.0, 0.0, 0.0], 'base_init_qpos': [0.0], 'num_sample_vertics': 300, 'convexify': False, 'fixed': False, 'actuated': True, 'kp': 80.0, 'kv': 5.0, 'force_range': 50.0, 'collect_data': False, 'offset_pos': [0.0, 0.0, 0.0], 'color': None, 'show_link_frame': False, 'urdf_path': r'E:\IBS\dexmachina\dexmachina\assets\arctic\mixer\decomp\mixer_decomp.urdf', 'bottom_mesh_fname': r'E:\IBS\dexmachina\dexmachina\assets\arctic\mixer\bottom_watertight_tiny.stl', 'top_mesh_fname': r'E:\IBS\dexmachina\dexmachina\assets\arctic\mixer\top_watertight_tiny.stl'}}   

    
    env_kwargs['robot_cfgs']['left']['urdf_path'] = r'E:\IBS\dexmachina\dexmachina\assets\inspire_hand\left_xyz_copy.urdf'
    env_kwargs['robot_cfgs']['right']['urdf_path'] = r'E:\IBS\dexmachina\dexmachina\assets\inspire_hand\right_xyz_copy.urdf'


    
    
    assert env_kwargs['env_cfg']['use_rl_games'], "The saved environment is not from rl-games"
    
    if args.raytrace and args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['raytrace'] = True

    env_kwargs['env_cfg']['early_reset_threshold'] = 0.0
    print("WARNING: setting env.is_eval to True")
    env_kwargs['env_cfg']['is_eval'] = True
    if args.vis:
        # NOTE still needs this because kwargs are loaded from saved env
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True
        env_kwargs['env_cfg']['scene_kwargs']['show_viewer'] = True
    print("Not applying external forces during eval time")
    env_kwargs['rand_cfg']['randomize'] = False 



    env_kwargs['env_cfg']['num_envs'] = args.num_envs
    env_kwargs['env_cfg']['rand_init_ratio'] = 0.0
    if args.record_video:
        env_kwargs['env_cfg']['scene_kwargs']['use_visualizer'] = True  
        env_kwargs['env_cfg']['record_video'] = True 
        print(f"Setting render resolution to 512") 
        # env_kwargs['camera_res'] = (2048, 2048)
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
    # =========================================================
    
    env_kwargs['demo_data'] = newdemo_data

    print("Removing any curriculum config during eval")
    env_kwargs.pop("curriculum_cfg") 
    device = torch.device('cuda:0')
    import genesis as gs
    gs.init(backend=gs.gpu, logging_level='warning')

    
    env = BaseEnv(
         **env_kwargs
    )
    scene= env.scene
    hand_entities = dict()

    hand_right= scene.add_entity( 
            gs.morphs.URDF(
                    file=r"E:\IBS\dexmachina\dexmachina\assets\mano_double\rhand_mano_low_mass.urdf",
                    pos=np.array([0, 0, 0]),
                    quat=(1, 0, 0, 0),
                    fixed=True,
                    convexify=True,
                    merge_fixed_links=False, # NOTE: need to keep this to track the fingertip links
                    recompute_inertia=True,
                    scale=URDFSCALE,
                    
                ),
                visualize_contact=True,
                material=gs.materials.Rigid(gravity_compensation=0.8),
                vis_mode="collision" if args.vis_collision else "visual",
        )
    hand_left= scene.add_entity( 
            gs.morphs.URDF(
                    file=r"E:\IBS\dexmachina\dexmachina\assets\mano_double\lhand_mano_low_mass.urdf",
                    pos=np.array([0, 0, 0]),
                    quat=( 1, 0, 0, 0),
                    fixed=True,
                    convexify=True,
                    merge_fixed_links=False, # NOTE: need to keep this to track the fingertip links
                    recompute_inertia=True,
                    scale=URDFSCALE,
                    
                ),
                visualize_contact=True,
                material=gs.materials.Rigid(gravity_compensation=0.8),
                vis_mode="collision" if args.vis_collision else "visual",
        )

    

    env.build_scene()
    env.post_scene_build_setup()
    hand_joint_vals = dict()


 
    scene.reset() 

   

    n_steps = args.interp_steps 


    # set the joint values and step the scene
    step = 0

    frames = [] 

    video_path = f"rendered/hands/{hand_names[-1]}"
    # os.makedirs(os.path.dirname(video_path), exist_ok=True)
    # frame_path = f"{video_path}/frames"
    # if args.raytrace:
    #     frame_path = f"{video_path}/raytrace"
    # # os.makedirs(frame_path, exist_ok=True) 
    # quat_combos = None
    # if args.iterate_quat:
    #     quat_combos = generate_90degree_rotation_quaternions()
    # set the camera z axis (both pos and lookat) to align with the hand's first link 


    # hand_right.set_dofs_position(
    #             position=[0]*51,
    #             dofs_idx_local= [i for i in range(51)],
    #             zero_velocity=True,
    #         ) 
    # for i in range(51):
    #     print("handjoint[0]:", hand.joints[i].pos)
    for hand in [hand_left]:
        all_joints = hand.joints
        actuated_joints = [joint for joint in all_joints if joint.type in [gs.JOINT_TYPE.REVOLUTE, gs.JOINT_TYPE.PRISMATIC]]

        act_idxs = [joint.dof_idx_local for joint in actuated_joints]
        for joint in actuated_joints:
            print(f"Joint {joint.name} dof: {joint.dof_idx_local} ")

    scene.step()
    export_dir = "exported_hand_meshes_zero"
    os.makedirs(export_dir, exist_ok=True)
        
    # # 导出每只手的网格
    # for name, hand in hand_entities.items():
    #     # 获取手部所有链接
    #     links = hand.links
        
    #     # 收集所有几何体的顶点和面
    #     all_vertices = []
    #     all_faces = []
    #     vertex_offset = 0
        
    #     for link in links:
    #         # 获取链接的世界坐标顶点
    #         link_vertices = link.get_vverts().detach().cpu().numpy()
            
    #         # 如果是批量环境，取第一个环境的数据
    #         if len(link_vertices.shape) == 3:
    #             link_vertices = link_vertices[0]
                
    #         # 获取链接的可视化几何体
    #         for vgeom in link.vgeoms:
    #             # 获取几何体的面数据
    #             faces = vgeom._init_vfaces
                
    #             # 添加顶点到总列表
    #             all_vertices.append(link_vertices)
                
    #             # 调整面的顶点索引
    #             adjusted_faces = faces + vertex_offset
    #             all_faces.append(adjusted_faces)
                
    #             # 更新顶点偏移
    #             vertex_offset += len(link_vertices)
        
    #     # 合并所有顶点和面
    #     if all_vertices and all_faces:
    #         combined_vertices = np.vstack(all_vertices)
    #         combined_faces = np.vstack(all_faces)
            
    #         # 创建trimesh对象并导出为OBJ文件
    #         mesh_obj = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces, process=False)
    #         obj_filename = os.path.join(export_dir, f"{name}_step_{step:04d}.obj")
    #         mesh_obj.export(obj_filename)





    
    
    scene.step()
    exportmesh=False



    with open(r"E:\IBS\dexmachina\arctic\data\body_models\mano\MANO_RIGHT.pkl", 'rb') as f:
        mano_data_right = pickle.load(f, encoding='latin1')
    hand_mean_pose_right = mano_data_right['hands_mean']
    with open(r"E:\IBS\dexmachina\arctic\data\body_models\mano\MANO_LEFT.pkl", 'rb') as f:
        mano_data_left = pickle.load(f, encoding='latin1')
    hand_mean_pose_left = mano_data_left['hands_mean']
    while True: 


        #interp_joints, act_idxs = hand_joint_vals[name]
        mano_ordered_act_idxs = [
            0, 1, 2, 3, 4, 5,  # 腕关节 (trans + rot)
            6, 11, 16, 21, 26, 31, 36, 41, 46,  # 拇指 (Thumb)
            7, 12, 17, 22, 27, 32, 37, 42, 47,    # 食指 (Index)
            8, 13, 18, 23, 28, 33, 38, 43, 48,   # 中指 (Middle)
            9, 14, 19, 24, 29, 34, 39, 44, 49,   # 无名指 (Ring)
            10, 15, 20, 25, 30, 35, 40, 45, 50,  # 小指 (Pinky)
        ]
        # mano_ordered_act_idxs = [
        #     0, 1, 2, 3, 4, 5,  # 腕关节 (trans + rot)
        #     8, 13, 18, 23, 28, 33, 38, 43, 48,  # 拇指 (Thumb)
        #     10, 15, 20, 25, 30, 35, 40, 45, 50,  # 食指 (Index)
        #     7, 12, 17, 22, 27, 32, 37, 42, 47,    # 中指 (Middle)
        #     6, 11, 16, 21, 26, 31, 36, 41, 46,  # 无名指 (Ring)

        #     9, 14, 19, 24, 29, 34, 39, 44, 49,    # 小指 (Pinky)
        # ]
        trans_data_right = hand_mano_data["right"]["trans"][step]-WRISTTRANS_OFFSET_RIGHT
        rot_data_right = hand_mano_data["right"]["rot"][step]      # 3维
        pose_data_right = hand_mano_data["right"]["pose"][step]+hand_mean_pose_right  # 45维
        correct_quaternion(pose_data_right) 
        pose_data_origin=hand_mano_data["right"]["pose"][step]
        
        print(f"hand_mean_pose={hand_mean_pose_right}")
        
        print(f"pose_data={ pose_data_origin}")   # 45维
        mano_joints_right = np.concatenate([trans_data_right, rot_data_right, pose_data_right])
        print(mano_joints_right)
        jpos_right = torch.tensor(mano_joints_right, dtype=torch.float32)[None].repeat(args.num_envs, 1)
        # print(f"Setting joint positions for {name}: {jpos}")
        hand_right.set_dofs_position(
            position=jpos_right,
            dofs_idx_local= mano_ordered_act_idxs,
            zero_velocity=True,
        ) 

        trans_data_left = hand_mano_data["left"]["trans"][step]-WRISTTRANS_OFFSET_LEFT
        rot_data_left = hand_mano_data["left"]["rot"][step]
        pose_data_left = hand_mano_data["left"]["pose"][step]+hand_mean_pose_left
        correct_quaternion(pose_data_left)
        mano_joints_left = np.concatenate([trans_data_left, rot_data_left, pose_data_left])
        print(mano_joints_right)
        jpos_left = torch.tensor(mano_joints_left, dtype=torch.float32)[None].repeat(args.num_envs, 1)
        hand_left.set_dofs_position(
            position=jpos_left,
            dofs_idx_local= mano_ordered_act_idxs,
            zero_velocity=True,
        )


            
        scene.step()
        if not exportmesh:
            # 导出手部网格为OBJ文件
            # 创建导出目录
            
            export_dir = "exported_hand_meshes_right"
            os.makedirs(export_dir, exist_ok=True)
            
            # 导出每只手的网格
           
                # 获取手部所有链接
            hand = hand_right
            name = "hand_right"
            links = hand.links
            
            # 收集所有几何体的顶点和面
            all_vertices = []
            all_faces = []
            vertex_offset = 0
            
            for link in links:
                # 获取链接的世界坐标顶点
                link_vertices = link.get_vverts().detach().cpu().numpy()
                
                # 如果是批量环境，取第一个环境的数据
                if len(link_vertices.shape) == 3:
                    link_vertices = link_vertices[0]
                    
                # 获取链接的可视化几何体
                for vgeom in link.vgeoms:
                    # 获取几何体的面数据
                    faces = vgeom._init_vfaces
                    
                    # 添加顶点到总列表
                    all_vertices.append(link_vertices)
                    
                    # 调整面的顶点索引
                    adjusted_faces = faces + vertex_offset
                    all_faces.append(adjusted_faces)
                    
                    # 更新顶点偏移
                    vertex_offset += len(link_vertices)
            
            # 合并所有顶点和面
            if all_vertices and all_faces:
                combined_vertices = np.vstack(all_vertices)
                combined_faces = np.vstack(all_faces)
                
                # 创建trimesh对象并导出为OBJ文件
                mesh_obj = trimesh.Trimesh(vertices=combined_vertices, faces=combined_faces, process=False)
                obj_filename = os.path.join(export_dir, f"{name}_step_{step:04d}.obj")
                mesh_obj.export(obj_filename)

        step += 1
        if step == maxstep:
            step = 0 
            exportmesh=True
            if args.record_video:
                export_video(frames, join(video_path, f"{hand_names[-1]}_render.mp4"), fps=10) 
                print(f"Exported video to {video_path}")  
                exit()

    breakpoint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--vis', '-v', action='store_true')
    parser.add_argument('--hand', type=str, default='inspire')
    # take in multiple hand names into a list of args
    parser.add_argument('--hands_list', type=str, nargs='+', default=['inspire'])
    parser.add_argument('--urdf_path', '-u', type=str, default='') # if provided, use this urdf path and vis only one hand
    parser.add_argument('--group_collisions', action='store_true')
    parser.add_argument('--side', type=str, default='left')
    parser.add_argument('--asset_path', type=str, default='/home/mandi/chiral/assets')
    parser.add_argument('--vis_collision', '-vc', action='store_true')
    parser.add_argument('--vis_6dof', action='store_true', help='Visualize the scene')
    parser.add_argument('--record_video', action='store_true', help='Render a video')
    parser.add_argument('--render_image', action='store_true', help='Render images')
    parser.add_argument('--raytrace', action='store_true', help='Use raytracer')
    parser.add_argument('--transparent', '-tr', action='store_true', help='Use transparent rendering')
    parser.add_argument('--interp_steps', type=int, default=50, help='Number of interpolation steps')
    parser.add_argument('--interp_joints_separately', action='store_true', help='Interpolate wrist and finger joints separately')
    parser.add_argument('--skip_wrist_interp', action='store_true', help='Skip interpolating wrist joints')
    parser.add_argument('--wrist_only', action='store_true', help='Only interpolate wrist joints')
    parser.add_argument('--wrist_rot_only', action='store_true', help='Only interpolate wrist rotation joints')
    parser.add_argument('--num_envs', type=int, default=1, help='Number of environments to build')
    parser.add_argument('--print_joints', action='store_true', help='Print joint names')
    parser.add_argument('--gather_geoms', action='store_true', help='Gather geoms')
    parser.add_argument('--base_link_name', type=str, default='base_link', help='Name of the base link')
    parser.add_argument('--enable_self_collision', action='store_true', help='Enable selfcollision')
    parser.add_argument('--iterate_quat', action='store_true', help='Iterate through quaternion combinations')
    args = parser.parse_args()

    main(args)


