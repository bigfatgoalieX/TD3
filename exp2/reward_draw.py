import numpy as np
import matplotlib.pyplot as plt

# 加载 .npy 文件
rewards = np.load("./results/TD3_Walker2d-v5_0.npy")  # 替换成你的文件路径

# 画图
plt.figure(figsize=(10, 5))
plt.plot(rewards, marker='o', linestyle='-', color='blue')
plt.title("Evaluation Reward Over Time")
plt.xlabel("Evaluation Step")
plt.ylabel("Average Reward")
plt.grid(True)
plt.tight_layout()
plt.show()
