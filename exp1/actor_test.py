import torch
import gymnasium as gym
import numpy as np
from TD3 import Actor  # 从你的 TD3.py 文件中引入 Actor 类

# ---------- 1. 新环境 ----------
env = gym.make("HalfCheetah-v5", render_mode='human')  # 你想测试的另一个环境
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

# ---------- 2. 初始化 actor 并加载参数 ----------
actor = Actor(state_dim, action_dim, max_action)
actor.load_state_dict(torch.load("./models/TD3_HalfCheetah-v5_0_dr_1_actor"))
actor.eval()  # 关闭 Dropout、BN 等

# ---------- 3. 运行测试 ----------
state, _ = env.reset()
done = False
total_reward = 0

while not done:
    state_tensor = torch.FloatTensor(state.reshape(1, -1))
    action = actor(state_tensor).detach().cpu().numpy().flatten()
    state, reward, terminated, truncated, _ = env.step(action)
    env.render()
    total_reward += reward
    done = terminated or truncated

print("Evaluation Reward in new environment:", total_reward)
