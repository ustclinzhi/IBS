import os
from os.path import join
from dexmachina.asset_utils import get_urdf_path

new_asset_dir = "mano_double/"
left_rel_urdf = join(new_asset_dir, "rhand_mano_low_mass.urdf")
right_rel_urdf = join(new_asset_dir, "rhand_mano_low_mass.urdf")

# 原来的 finger-structured mimic 关系在 new_hand URDF 中不可映射（关节/链接命名不同）
# 所以这里将 mimic map 设为空，交由上层 demo/data 在需要时填写
NEW_LEFT_MIMIC_JOINTS = {}
NEW_RIGHT_MIMIC_JOINTS = {}

# new_hand URDF 中的顶级手掌链接名为 "hand"，手指子链接为 link_joint_0 .. link_joint_15
# 这里选择把 5 个关键点映射为 URDF 中的 5 个 link_joint_*（视作 fingertip 占位）
# 如果需要不同关键点，可在上层 demo 数据中重写 kpt_link_names

# URDF 中存在 16 个 revolute joints: joint_hand_to_link_joint_0 .. joint_hand_to_link_joint_15
# default_qpos 长度应与实际自由度数一致，这里默认全部 0（可被 demo 数据重写）
DEFAULT_DOFS = 16
NEW_DEFAULT_QPOS = {
    "left": [0.0] * DEFAULT_DOFS,
    "right": [0.0] * DEFAULT_DOFS,
}

NEW_LEFT_CFG = {
    "urdf_path": get_urdf_path(left_rel_urdf),
    "wrist_link_name": "right_wrist_rz",                    # 对应 URDF 中的 palm link（参考 DEX3 的 left_hand_palm_link）
    "mimic_joint_map": NEW_LEFT_MIMIC_JOINTS.copy(),
    # Add palm link first (匹配 DEX3 要求：palm 在关键点列表中)，其余为 fingertip 占位，可用 inspect 脚本替换为 collision 链接名
    "kpt_link_names":['right_wrist_rz', 'right_index1_z', 'right_middle1_z', 'right_pinky1_z', 'right_ring1_z', 'right_thumb1_z', 'right_index2_z', 'right_middle2_z', 'right_pinky2_z', 'right_ring2_z', 'right_thumb2_z', 'right_index3_z', 'right_middle3_z', 'right_pinky3_z', 'right_ring3_z', 'right_thumb3_z'],
    # NOTE: 可能在 demo 数据中被重写
    # collision_groups 需由 "hand_proc/inspect_raw_urdf.py --gather_geoms" 输出确定，这里填入示例占位，运行 inspect 后请替换为准确映射
    "joint_limits": {},
    "default_qpos": NEW_DEFAULT_QPOS["left"],
    "actuators": {
        # 参考 DEX3，先把占位的 actuator 参数放好；finger 匹配 new_hand 的 joint_hand_to_link_joint_<n>
        "finger": dict(
            joint_exprs=[],
            kp=20.0,
            kv=1.5,               # 调整为 DEX3 建议的 kv
            force_range=50.0,
        ),
        "wrist_rot": dict(
            joint_exprs=[],      # new_hand 无 forearm joints，留空；若有 forearm，请替换为具体 regex
            kp=80,
            kv=5.0,
            force_range=50.0,
        ),
        "wrist_trans": dict(
            joint_exprs=[],
            kp=350,
            kv=12.0,
            force_range=50.0,
        ),
    },
    # 示例 collision_groups（必须用 inspect 脚本确认并替换）
    "collision_groups": {7: 0, 23: 1, 24: 2, 25: 3, 26: 4, 27: 5, 38: 1, 39: 2, 40: 3, 41: 4, 42: 5, 53: 1, 54: 2, 55: 3, 56: 4, 57: 5},
    "collision_palm_name": "right_wrist_rz",
}

NEW_RIGHT_CFG = {
    "urdf_path": get_urdf_path(right_rel_urdf),
    "wrist_link_name": "hand",
    "mimic_joint_map": NEW_RIGHT_MIMIC_JOINTS.copy(),
    "kpt_link_names":['hand', 'link_joint_0', 'link_joint_1', 'link_joint_2', 'link_joint_3', 'link_joint_4', 'link_joint_5', 'link_joint_6', 'link_joint_7', 'link_joint_8', 'link_joint_9', 'link_joint_10', 'link_joint_11', 'link_joint_12', 'link_joint_13', 'link_joint_14', 'link_joint_15'],
    # NOTE: 可能在 demo 数据中被重写
    "joint_limits": {},
    "default_qpos": NEW_DEFAULT_QPOS["right"],
    # 复用 left 的 actuator 配置（对称手模型通常可复用）
    "actuators": NEW_LEFT_CFG["actuators"].copy(),
    "collision_groups": {7: 0},
    "collision_palm_name": NEW_LEFT_CFG["collision_palm_name"],
}

NEW_CFGs = dict(left=NEW_LEFT_CFG, right=NEW_RIGHT_CFG)

#python hand_proc/inspect_raw_urdf.py --gather_geoms --num_envs 1 --urdf_path assets/mano_double/rhand_mano_low_mass.urdf --base_link_name hand
#python hand_proc/inspect_raw_urdf.py --urdf_path assets/new_hand/right/mano_hand_right_6dof.urdf -v
#python hand_proc/inspect_raw_urdf.py --urdf_path assets/mano_double/lhand_mano_low_mass.urdf -v --interp_steps 500

#python hand_proc/inspect_raw_urdf.py --urdf_path assets/inspire_hand/left_xyz_copy.urdf -v

#python hand_proc/add_wrist_dof.py assets/new_hand/right/mano_hand_right_multi.urdf mano_hand_right_6dof.urdf --dof_choices all --base_link hand
#python hand_proc/minimal_retarget.py --clip box-30-230 --hand new_hand -vis
#python retargeting/parallel_retarget.py --clip box-30-230 --hand new_hand --control_steps 2000 --save_name para --save -ow 