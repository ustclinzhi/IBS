import sys
import os
import torch
import numpy as np

# 添加仓库路径
sys.path.append("E:/IBS/dexmachina/Pointnet_Pointnet2_pytorch-master/Pointnet_Pointnet2_pytorch-master")

from models.pointnet2_cls_ssg import get_model

# 初始化 PointNet++ SSG 模型（40类，无 normal）
model = get_model(num_class=40, normal_channel=False)

# 加载预训练权重
checkpoint_path = r"E:\IBS\dexmachina\Pointnet_Pointnet2_pytorch-master\Pointnet_Pointnet2_pytorch-master\log\classification\pointnet2_ssg_wo_normals\checkpoints\best_model.pth"
checkpoint = torch.load(checkpoint_path, map_location='cpu')
model.load_state_dict(checkpoint['model_state_dict'])

model.eval()
print("✅ 预训练 PointNet++ 模型加载成功！")

# 准备点云: (1, 1024, 3)
point_cloud = np.random.randn(1024, 3)
point_cloud = point_cloud - np.mean(point_cloud, axis=0)
dist = np.max(np.sqrt(np.sum(point_cloud**2, axis=1)))
point_cloud = point_cloud / dist
pc_tensor = torch.from_numpy(point_cloud).float().unsqueeze(0)  # (1, 1024, 3)
pc_tensor = pc_tensor.transpose(2, 1)
# 提取特征
with torch.no_grad():
    # 现在输入张量形状为(batch_size, 3, num_points)
    logits, l_features = model(pc_tensor)  # 只传递点云张量

    # 根据模型定义，l_features是全局特征 (l3_points)
    global_feat = l_features  # 获取特征
    
    # 确保特征向量形状是 (1, 1024) 用于后续使用
    if len(global_feat.shape) == 2 and global_feat.shape[0] == 1:
        # 形状已经是 (1, 1024)
        feat_vector = global_feat
    elif len(global_feat.shape) == 2 and global_feat.shape[1] == 1:
        # 形状是 (1024, 1)，需要转置
        feat_vector = global_feat.transpose(0, 1)
    else:
        # 其他情况尝试 reshape
        feat_vector = global_feat.view(1, -1)

print("✅ 特征向量形状:", feat_vector.shape)
print("✅ 前5维特征:", feat_vector[0][:min(5, feat_vector.shape[1])])