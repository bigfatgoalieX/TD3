import numpy as np
import torch
import cma
import gymnasium as gym
import argparse

from exp2.TD3 import Actor
from exp2.utils import TargetDomainWrapper
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3PolicyWrapper:
    def __init__(self, model_path, state_dim, action_dim, max_action, mu_dim):
        self.actor = Actor(state_dim, mu_dim, action_dim, max_action)
        self.actor.load_state_dict(torch.load(model_path))
        self.actor.eval()

    def predict(self, obs, mu_prime):
        state = torch.FloatTensor(obs.reshape(1, -1)).to(device)
        mu = torch.FloatTensor(mu_prime.reshape(1, -1)).to(device)
        return self.actor(state, mu).cpu().data.numpy().flatten()


def make_target_env(mu_target):
    """
    返回一个使用固定 mu_target 的环境（不再变化）
    """
    env = gym.make("HalfCheetah-v5")
    wrapped_env = TargetDomainWrapper(
        env,
        gravity_scale=mu_target[0],
        friction_scale=mu_target[1],
        mass_scale=mu_target[2]
    )
    return wrapped_env


def evaluate_policy_with_mu_prime(mu_prime, policy, env, num_episodes=5):
    """
    在给定固定目标环境上，用 mu' 配置策略，评估其平均回报
    """
    returns = []
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = policy.predict(obs, mu_prime)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        returns.append(total_reward)
    return np.mean(returns)


def optimize_mu_prime(policy, env, mu_dim, sigma=0.5, max_iter=30):
    """
    优化策略的输入配置向量 mu'，使其在目标环境上表现最优
    """
    mu_prime_init = np.zeros(mu_dim)
    es = cma.CMAEvolutionStrategy(mu_prime_init, sigma)

    for generation in range(max_iter):
        solutions = es.ask()
        fitnesses = [-evaluate_policy_with_mu_prime(mu_prime, policy, env) for mu_prime in solutions]
        es.tell(solutions, fitnesses)
        es.logger.add()
        print(f"[Gen {generation}] Best reward: {-min(fitnesses):.2f}")

    best_mu_prime = es.result.xbest
    print("\n Optimization finished. Best mu' found:", best_mu_prime)
    return best_mu_prime


if __name__ == "__main__":
    # === 设定目标环境参数 ===
    parser = argparse.ArgumentParser()
    parser.add_argument("--gravity", type=float, default=1.0)
    parser.add_argument("--friction", type=float, default=1.0)
    parser.add_argument("--mass", type=float, default=1.0)
    args = parser.parse_args()
    
    mu_target = np.array([args.gravity, args.friction, args.mass])  # 假设我们知道的目标环境物理属性

    MODEL_PATH = "exp2/models/TD3_HalfCheetah-v5_0_3_actor"
    state_dim = 17
    action_dim = 6
    mu_dim = 3
    max_action = 1.0

    # 初始化策略和环境
    policy = TD3PolicyWrapper(MODEL_PATH, state_dim, action_dim, max_action, mu_dim)
    target_env = make_target_env(mu_target)

    # 优化策略输入配置 mu'
    best_mu_prime = optimize_mu_prime(policy, target_env, mu_dim, sigma=0.5, max_iter=30)
