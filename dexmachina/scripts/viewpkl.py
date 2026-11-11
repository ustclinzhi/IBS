import pickle
import sys
if sys.platform == "win32":
    import pathlib
    pathlib.PosixPath = pathlib.WindowsPath
saved_cfg_fname = r"E:\IBS\dexmachina\dexmachinatest\inspire_hand12.1test\inspire-test_waffleiron40-340-s01-u01_B12000_hybrid_thres0.5_ho16_imi0.2_con2.0_bc0.2_ibsbeta25.0\params\env.pkl"
with open(saved_cfg_fname, "rb") as f:
    cfg = pickle.load(f)
    print(cfg['robot_cfgs'])



#python scripts/viewpkl.py 
"""
{'left': {'name': 'inspire_hand_left', 'gravity_compensation': 0.8, 'actuators': {'finger': {'joint_exprs': ['.*J1', '.*J2', '.*J3', '.*J4'], 'kp': 20.0, 'kv': 1.0, 'fo ('L_FF_J1', 1.13), 'L_MF_J2': ('L_MF_J1', 1.13), 'L_RF_J2': ('L_RF_J1', 1.08), 'L_LF_J2': ('L_LF_J1', 1.13)}, 'joint_limits': {'L_forearm_tx_link_joint': (-0.05, 0.3), 'L_forearm_ty_link_joint': (-0.1, 0.2), 'L_forearm_tz_link_joint': (1.0, 1.3), 'L_forearm_roll_link_joint': (-0.25, 0.5), 'L_forearm_pitch_link_joint': (0.2, 0.6), 'L_forearm_yaw_link_joint': (-0.7, 0.4)}, 'default_qpos': [0.25, 0.11, 1.0, 0.76, 0.18, 0.58, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'collision_groups': {7: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 1, 23: 1}, 'collision_palm_name': 'base_link'}, 'right': {'name': 'inspire_hand_right', 'gravity_compensation': 0.8, 'actuators': {'finger': {'joint_exprs': ['.*J1', '.*J2', '.*J3', '.*J4'], 'kp': 20.0, 'kv': 1.0, 'force_range': 50.0}, 'wrist_rot': {'joint_exprs': ['[LR]_forearm_(roll|pitch|yaw)_link_joint'], 'kp': 80, 'kv': 5.0, 'force_range': 50.0}, 'wrist_trans': {'joint_exprs': ['[LR]_forearm_t[xyz]_link_joint'], 'kp': 350, 'kv': 12.0, 'force_range': 50.0}}, 'base_init_pos': [0.0, 0.0, 0.0], 'base_init_quat': [1.0, 0.0, 0.0, 0.0], 'action_moving_avg': 1.0, 'action_mode': 'hybrid', 'wrist_only': False, 'kpt_link_names': ['R_TH_Tip', 'R_FF_Tip', 'R_MF_Tip', 'R_RF_Tip', 'R_LF_Tip', 'base_link', 'R_TH_L1', 'R_FF_L1', 'R_MF_L1', 'R_RF_L1', 'R_LF_L1', 'R_TH_L2', 'R_FF_L2', 'R_MF_L2', 'R_RF_L2', 'R_LF_L2', 'R_TH_L3', 'R_TH_L4'], 'collect_data': False, 'use_saved_targets': False, 'hybrid_scales': (0.1, 1.0), 'res_cap': False, 'show_keypoints': False, 'visualization': True, 'urdf_path': WindowsPath('/public/home/sa24001033/dexmachina/dexmachina/assets/inspire_hand/right_xyz_copy.urdf'), 'wrist_link_name': 'base_link', 'mimic_joint_map': {'R_TH_J3': ('R_TH_J2', 1.0), 'R_TH_J4': ('R_TH_J2', 1.13), 'R_FF_J2': ('R_FF_J1', 1.13), 'R_MF_J2': ('R_MF_J1', 1.13), 'R_RF_J2': ('R_RF_J1', 1.08), 'R_LF_J2': ('R_LF_J1', 1.13)}, 'joint_limits': {'R_forearm_tx_link_joint': (-0.4, -0.1), 'R_forearm_ty_link_joint': (-0.1, 0.2), 'R_forearm_tz_link_joint': (1.0, 1.3), 'R_forearm_roll_link_joint': (-0.4, 0.2), 'R_forearm_pitch_link_joint': (0.0, 0.6), 'R_forearm_yaw_link_joint': (2.5, 3.1)}, 'default_qpos': [-0.35, 0.1, 1.05, -0.37, 0.23, 2.64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'collision_groups': {7: 0, 13: 1, 14: 2, 15: 3, 16: 4, 17: 5, 18: 1, 23: 1}, 'collision_palm_name': 'base_link'}}
"""