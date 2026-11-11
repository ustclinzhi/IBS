# =====================================================================
#  IBS-DexMachina
#  交互平分面 (Interaction Bisector Surface, IBS) 引导的灵巧手功能性重定向
#
#  基于 DexMachina (Zhao Mandi et al., arXiv:2505.24853) 二次开发。
#  本仓库只包含源码、配置与小尺寸资产；数据集、训练权重、
#  第三方大型依赖（Genesis / rl_games / pytorch3d / ARCTIC）请按下文自行安装。
# =====================================================================

本仓库在 [DexMachina](https://github.com/MandiZhao/dexmachina) 框架上增加了
**IBS（交互平分面）** 表征，用于灵巧手的功能性重定向。核心改动集中在：

| 文件 | 作用 |
| --- | --- |
| `computibs.py` | `DifferentiableIBS`：IBS 求解器（点云 + KNN 最近邻 + 迭代优化 + 有符号距离），基于 `pytorch3d.ops.knn_points` |
| `getibs.py` | 离线批量生成演示 IBS（4096 采样 / 60 次迭代） |
| `viewibs.py`、`dexmachina/envs/viewibs.py` | IBS 求解器的可视化 / 调试版本 |
| `dexmachina/envs/rewards.py` | `compute_ibs_reward()`：手部点云与参考 IBS 的 Chamfer 距离 → `exp(-β·CD)` 奖励 |
| `dexmachina/envs/base_env.py` | 环境观测拼接、`_get_rewards` 中手部点云生成（`process_robot_links_fast`） |
| `dexmachina/rl/train_rl_games.py` | 新增 `--ibs` / `--ibsbeta` 参数并注入 `reward_cfg` |
| `dexmachina/rl/eval_rl_games.py` | 评测脚本，逐帧导出 `eval_ep*.npy`（obj_state / demo_state / pos_dist / rot_dist / arti_dist） |

---

## 一、环境准备

先按上游 DexMachina 的流程装好依赖（Python 3.10 + torch 2.5.1）：

```bash
conda create -n dexmachina python=3.10 -y && conda activate dexmachina
pip install torch==2.5.1

git clone https://github.com/MandiZhao/Genesis.git
cd Genesis && pip install -e . && pip install libigl==2.5.1 && cd ..

git clone https://github.com/MandiZhao/rl_games.git
cd rl_games && pip install -e . && cd ..

pip install gymnasium ray seaborn wandb trimesh
```

本仓库额外需要：

```bash
pip install pytorch3d        # IBS 求解器依赖 knn_points
pip install open3d           # 点云 / 网格处理与可视化
```

安装本项目：

```bash
pip install -e .
```

> **路径提示**：`dexmachina/envs/rewards.py` 中通过 `from viewibs import DifferentiableIBS`
> 导入，因此需要把项目根目录加入 `PYTHONPATH`（或始终在根目录下运行脚本）。

---

## 二、数据与资产（不随仓库分发）

以下内容因体积或许可证原因未纳入版本管理，需自行获取并放到对应位置：

| 路径 | 内容 | 获取方式 |
| --- | --- | --- |
| `arctic/` | ARCTIC 数据集与处理脚本 | https://github.com/zc-alexfan/arctic |
| `arctic/data/body_models/` | MANO / SMPL-X 模型 | 需在 https://mano.is.tue.mpg.de 申请许可后下载 |
| `dexmachina/hand_proc/retargeted/` | 重定向后的 `.pt` 演示数据 | 运行 `dexmachina/retargeting/parallel_retarget.py` 生成 |
| `dexmachina/assets/arctic/` | 物体网格（box / ketchup / mixer / notebook / waffleiron） | ARCTIC 仓库 |
| `demoibs_*.npy`、`*_kpt_data.npy` | 关键点与 IBS 缓存 | 运行 `getibs.py` 及 `dexmachina/scripts/*` 生成 |

---

## 三、训练与评测

生成演示 IBS 缓存：

```bash
python getibs.py
```

训练（脚本默认开启 IBS 奖励）：

```bash
bash examples/train_rl.sh
```

`train_rl.sh` 中的关键参数：

| 参数 | 说明 |
| --- | --- |
| `-B` / `--num_envs` | 并行环境数 |
| `--ibs` | IBS 奖励权重，`0.0` 即关闭 |
| `--ibsbeta` | IBS 奖励的指数系数 β，默认 `20.0` |
| `--max_epochs` | 训练轮数 |

评测：

```bash
python -m dexmachina.rl.eval_rl_games --help
```

评测结果写入 `eval_ep*.npy`，包含逐帧的物体状态、演示状态以及位置 / 旋转 / 关节误差。

---

## 四、与上游的差异

相比上游 DexMachina，主要改动为：

1. **新增 IBS 表征与奖励**（上游完全没有）：`computibs.py`、`getibs.py`、`output_ibs.py`、`viewibs.py`、`rewards.py::compute_ibs_reward`。
2. **手部点云观测**：`base_env.py` 新增基于 link 顶点的快速采样（`process_robot_links_fast`）。
3. **训练脚本参数化**：`train_rl_games.py` 增加 `--ibs` / `--ibsbeta`，并由 `reward_cfg` 传入
   `RewardModule`，不再硬编码。
4. **新增评测脚本**：`eval_rl_games.py`、`eval_mano_grasp.py`。
5. **新增灵巧手配置**：`dexmachina/envs/hand_cfgs/inspire.py`、`new.py`、`dex3.py` 等。

> **关于"可微"**：IBS 求解器本身不含可学习参数，其输出仅作为观测 / 奖励的数值输入。
> PPO 的策略梯度只通过 `log πθ(a|s)` 回传策略与价值网络，因此"可微"是 PyTorch
> 实现带来的副产品，**不是**训练所必需的性质。这一设计使 IBS 可以在 GPU 上
> 在线实时计算并参与奖励塑形，而无需将求解器纳入梯度的反传路径。

---

## 五、引用

如果本代码对你的工作有帮助，请引用上游工作：

```bibtex
@misc{mandi2025dexmachinafunctionalretargetingbimanual,
      title={DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation},
      author={Zhao Mandi and Yifan Hou and Dieter Fox and Yashraj Narang and Ajay Mandlekar and Shuran Song},
      year={2025},
      eprint={2505.24853},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2505.24853},
}
```

## 六、许可

本仓库中来自 DexMachina 的部分沿用其原始许可（见 `LICENSE`），新增代码仅用于学术研究。
ARCTIC 数据、MANO / SMPL-X 模型、Genesis / rl_games / pytorch3d 等第三方组件
均遵循各自许可且**不在本仓库中分发**，使用者需自行申请并遵守。
