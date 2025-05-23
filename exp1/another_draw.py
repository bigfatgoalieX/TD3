import numpy as np
import matplotlib.pyplot as plt
import os

# 可选：使用 Seaborn 风格，提升图表美观
plt.style.use('seaborn-v0_8-darkgrid')  # 或 'seaborn-v0_8-darkgrid'，取决于你的 matplotlib 版本

# ----------- 参数设置 ------------
# 文件夹路径（保存你的 .npy 文件）
result_dir = "./results"
# 要加载的文件名（可修改）
file_names = [
    "TD3_HalfCheetah-v5_0_dr_4.npy",
    "TD3_HalfCheetah-v5_0_dr_1.npy",
    "TD3_HalfCheetah-v5_0_dr_3.npy",
    "TD3_HalfCheetah-v5_0_dr_2.npy",
]
# 每条曲线的标签（图例用）
# a lot of confusion here
# 顺序乱了...
labels = [
    "env0",
    "env1",
    "env2",
    "env3"
]
# 曲线颜色（可自定义）
colors = ["royalblue", "darkorange", "seagreen", "purple"]

# ----------- 平滑函数 ------------
def smooth(data, window_size=10):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# ----------- 开始画图 ------------
plt.figure(figsize=(12, 6))

for fname, label, color in zip(file_names, labels, colors):
    path = os.path.join(result_dir, fname)
    rewards = np.load(path)
    smoothed = smooth(rewards, window_size=10)
    
    # 原始曲线（淡色）
    plt.plot(rewards, color=color, alpha=0.3, linewidth=1)
    # 平滑后的曲线
    plt.plot(smoothed, label=label, color=color, linewidth=2)

# 图像标题与标签
plt.title("Training with Domain Randomization in 4 different Environments", fontsize=16, fontweight='bold')
plt.xlabel("Evaluation Step", fontsize=14)
plt.ylabel("Average Reward", fontsize=14)

# 网格与图例
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# 紧凑布局
plt.tight_layout()

# 显示图像
plt.show()
