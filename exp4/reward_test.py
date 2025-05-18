import torch
import gymnasium as gym
import numpy as np
from TD3 import Actor  # 确保你的 TD3.py 文件中有 Actor 类定义
import argparse
import utils

# 设置你保存的模型路径和测试的环境名称
MODEL_PATH = "./models/TD3_HalfCheetah-v5_0_dr_3_actor"
ENV_NAME = "HalfCheetah-v5"
NUM_EPISODES = 10

def evaluate_actor(actor, env, episodes=10):
    
    total_rewards = []

    for ep in range(episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state.reshape(1, -1))
            action = actor(state_tensor).detach().cpu().numpy().flatten()
            state, reward, terminated, truncated, _ = env.step(action)
            episode_reward += reward
            done = terminated or truncated

        print(f"Episode {ep+1} Reward: {episode_reward:.2f}")
        total_rewards.append(episode_reward)

    avg_reward = np.mean(total_rewards)
    print("---------------------------------------")
    print(f"Average Reward over {episodes} episodes: {avg_reward:.2f}")
    print("---------------------------------------")
    return avg_reward

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gravity",type = float, default = 1.0)
    parser.add_argument("--friction",type = float, default = 1.0)
    parser.add_argument("--mass",type = float, default = 1.0)
    parser.add_argument("--obs_noise",type = float, default = 0.0)
    parser.add_argument("--action_noise",type = float, default = 0.0)
    args = parser.parse_args()
    
    # 设置目标域环境
    target_env = gym.make("HalfCheetah-v5")
    target_env = utils.TargetDomainWrapper(
        target_env,
        gravity_scale = args.gravity,
        friction_scale = args.friction,
        mass_scale = args.mass,
        obs_noise_std = args.obs_noise,
        action_noise_std = args.action_noise
        )
    
    # 创建测试环境用于获取维度信息
    tmp_env = gym.make(ENV_NAME)
    state_dim = tmp_env.observation_space.shape[0]
    action_dim = tmp_env.action_space.shape[0]
    max_action = float(tmp_env.action_space.high[0])

    # 初始化 actor 并加载训练好的参数
    actor = Actor(state_dim, action_dim, max_action)
    actor.load_state_dict(torch.load(MODEL_PATH))
    actor.eval()

    # 跑 10 个 episode 并评估表现
    evaluate_actor(actor, target_env, NUM_EPISODES)
