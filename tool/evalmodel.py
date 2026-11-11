import subprocess
import os

# 替换为你的实际检查点路径
checkpoint_path1=r"E:\IBS\dexmachina\dexmachinatest12.22\inspire_hand\inspire-test_notebook40-340-s02-u02_B12000_hybrid_thres0.5_ho16_imi0.2_con2.0_bc0.2_ibsbeta25.0\nn\last_inspire_hand_ep_5000_rew__60.442707_.pth"
checkpoint_path2=r"E:\IBS\dexmachina\dexmachinatest12.22\inspire_hand\inspire-test_notebook40-340-s02-u02_B12000_hybrid_thres0.5_ho16_imi0.2_con2.0_bc0.2_ibsbeta25.0\nn\last_inspire_hand_ep_5000_rew__69.52295_.pth"
checkpoint_path3=r"E:\IBS\dexmachina\dexmachinatest12.22\inspire_hand\inspire-test_notebook40-340-s02-u02_B12000_hybrid_thres0.5_ho16_imi0.2_con2.0_bc0.2_ibsbeta25.0\nn\last_inspire_hand_ep_5000_rew__71.84007_.pth"
checkpoint_path4=r"E:\IBS\dexmachina\dexmachinatest12.22\inspire_hand\inspire-test_notebook40-340-s02-u02_B12000_hybrid_thres0.5_ho16_imi0.2_con2.0_bc0.2_ibsbeta25.0\nn\last_inspire_hand_ep_5000_rew__115.71418_.pth"
checkpoint_path5=r"E:\IBS\dexmachina\dexmachinatest12.22\inspire_hand\inspire-test_notebook40-340-s02-u02_B12000_hybrid_thres0.5_ho16_imi0.2_con2.0_bc0.2_ibsbeta25.0\nn\last_inspire_hand_ep_5000_rew__184.7671_.pth"

for path in [checkpoint_path1, checkpoint_path2, checkpoint_path3, checkpoint_path4, checkpoint_path5]:
    cmd = [
        "python", "dexmachina/rl/eval_rl_games.py",
        "--checkpoint", path,
       
        "-B", "1",
        # 添加其他你需要的参数
    ]
    
  
    subprocess.run(cmd)
