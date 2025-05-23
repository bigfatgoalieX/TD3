import numpy as np
import matplotlib.pyplot as plt

# 加载 .npy 文件
rewards = np.load("./results/TD3_HalfCheetah-v5_0_1.npy")  # 替换为实际路径

# 平滑处理（可选）: 使用滑动平均平滑曲线
def smooth(data, window_size=10):
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# 设置图像风格
plt.style.use('seaborn-v0_8-darkgrid')  # 可试试 'ggplot', 'bmh', 'fivethirtyeight' 等

# 创建图像
plt.figure(figsize=(12, 6))

# 平滑后的数据
smoothed_rewards = smooth(rewards, window_size=10)
plt.plot(smoothed_rewards, label='Smoothed Reward', color='royalblue', linewidth=2.0)

# 可选：原始数据（淡一点）
plt.plot(rewards, alpha=0.5, label='Raw Reward', color='silver', linewidth=1)

# 图像标题与标签
plt.title("TD3 on HalfCheetah-v5: Evaluation Reward Over Time", fontsize=16, fontweight='bold')
plt.xlabel("Evaluation Step", fontsize=14)
plt.ylabel("Average Reward", fontsize=14)

# 网格与图例
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend(fontsize=12)

# 紧凑布局
plt.tight_layout()

# 显示图像
plt.show()
