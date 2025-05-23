import numpy as np
import matplotlib.pyplot as plt
import os

# 设置美观的风格（确保你的 matplotlib 支持这个风格）
plt.style.use('seaborn-v0_8-darkgrid')

# ----------- 文件路径与配置 ------------
# 绝对路径的 .npy 文件（替换成你自己的绝对路径）
file_paths = [
    "../exp2/results/TD3_HalfCheetah-v5_0_4.npy",  # 对应 parameter_dimension=3
    "results/TD3_HalfCheetah-v5_0_highdim_1.npy",  # 对应 parameter_dimension=10
]

# 每条曲线的标签（图例）
labels = [
    "parameter_dimension = 3",
    "parameter_dimension = 10",
]

# 每条曲线的颜色
colors = [
    "royalblue",
    "darkorange",
]

# ----------- 平滑函数 ------------
def smooth(data, window_size=10):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# ----------- 绘图 ------------
plt.figure(figsize=(12, 6))

for path, label, color in zip(file_paths, labels, colors):
    rewards = np.load(path)
    smoothed = smooth(rewards, window_size=10)

    # 原始曲线（透明度较低）
    plt.plot(rewards, color=color, alpha=0.3, linewidth=1)
    # 平滑曲线（突出显示）
    plt.plot(smoothed, label=label, color=color, linewidth=2)

# 图像标题与轴标签
plt.title("Training with Conditional Policy Transfer while Parameter Dimension Changes", fontsize=16, fontweight='bold')
plt.xlabel("Evaluation Step", fontsize=14)
plt.ylabel("Average Reward", fontsize=14)

# 图例和网格
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.6)

# 布局优化
plt.tight_layout()

# 显示图像
plt.show()
