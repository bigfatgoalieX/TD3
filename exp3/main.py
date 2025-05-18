import numpy as np
import torch
import gymnasium as gym
import argparse
import os

import utils
import TD3
import OurDDPG
import DDPG


# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
    eval_env = gym.make(env_name)
    eval_env.reset(seed=seed + 100)
    
    # 需要获得mu的值来传入，从而进行测试
    mu = np.array([1.0, 1.0, 1.0])
    # 在这个eval_env里，暂且将mu固定
    
    avg_reward = 0.
    for _ in range(eval_episodes):
        state, _ = eval_env.reset()
        done = False
        while not done:
            action = policy.select_action(np.array(state), mu)
            state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            avg_reward += reward

    avg_reward /= eval_episodes

    print("---------------------------------------")
    print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
    print("---------------------------------------")
    return avg_reward


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="TD3")  # Policy name (TD3, DDPG or OurDDPG)
    parser.add_argument("--env", default="HalfCheetah-v5")  # OpenAI gym environment name
    parser.add_argument("--seed", default=0, type=int)  # Sets Gym, PyTorch and Numpy seeds
    parser.add_argument("--start_timesteps", default=25e3, type=int)  # Time steps initial random policy is used
    parser.add_argument("--eval_freq", default=5e3, type=int)  # How often (time steps) we evaluate
    parser.add_argument("--max_timesteps", default=1e6, type=int)  # Max time steps to run environment
    parser.add_argument("--expl_noise", default=0.1, type=float)  # Std of Gaussian exploration noise
    parser.add_argument("--batch_size", default=256, type=int)  # Batch size for both actor and critic
    parser.add_argument("--discount", default=0.99, type=float)  # Discount factor
    parser.add_argument("--tau", default=0.005, type=float)  # Target network update rate
    parser.add_argument("--policy_noise", default=0.2)  # Noise added to target policy during critic update
    parser.add_argument("--noise_clip", default=0.5)  # Range to clip target policy noise
    parser.add_argument("--policy_freq", default=2, type=int)  # Frequency of delayed policy updates
    parser.add_argument("--save_model", action="store_true")  # Save model and optimizer parameters
    parser.add_argument("--load_model", default="")  # Model load file name, "" doesn't load, "default" uses file_name
    args = parser.parse_args()

    attempt = 4
    file_name = f"{args.policy}_{args.env}_{args.seed}_{attempt}"
    print("---------------------------------------")
    print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}, Attempt: {attempt}")
    print("---------------------------------------")

    if not os.path.exists("./results"):
        os.makedirs("./results")
    if args.save_model and not os.path.exists("./models"):
        os.makedirs("./models")

    env = gym.make(args.env)
    env.reset(seed=args.seed)
    if hasattr(env.action_space, "seed"):
        env.action_space.seed(args.seed)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    state_dim = env.observation_space.shape[0]
    mu_dim = 3
    action_dim = env.action_space.shape[0]
    max_action = float(env.action_space.high[0])

    kwargs = {
        "state_dim": state_dim,
        "mu_dim": mu_dim,
        "action_dim": action_dim,
        "max_action": max_action,
        "discount": args.discount,
        "tau": args.tau,
    }

    # Initialize policy
    if args.policy == "TD3":
        kwargs["policy_noise"] = args.policy_noise * max_action
        kwargs["noise_clip"] = args.noise_clip * max_action
        kwargs["policy_freq"] = args.policy_freq
        policy = TD3.TD3(**kwargs)
    elif args.policy == "OurDDPG":
        policy = OurDDPG.DDPG(**kwargs)
    elif args.policy == "DDPG":
        policy = DDPG.DDPG(**kwargs)

    if args.load_model != "":
        policy_file = file_name if args.load_model == "default" else args.load_model
        policy.load(f"./models/{policy_file}")

    replay_buffer = utils.ReplayBuffer(state_dim, mu_dim, action_dim)

    evaluations = [eval_policy(policy, args.env, args.seed)]
    
    param_ranges = {
        "gravity_scale": [0.5, 1.5],
        "friction_scale": [0.5, 2.5],
        "mass_scale": [0.6, 1.4],
        "gear_scale": [0.8, 1.2],
    }
    randomizer = utils.PhysicsRandomizer(param_ranges)
    gravity_scale, friction_scale, mass_scale, gear_scale, mu = randomizer.sample()
    env = utils.TargetDomainWrapper(
        env,
        gravity_scale=gravity_scale,
        friction_scale=friction_scale,
        mass_scale=mass_scale,
        gear_scale=gear_scale,
    )
    
    state, _ = env.reset()
    episode_reward = 0
    episode_timesteps = 0
    episode_num = 0

    for t in range(int(args.max_timesteps)):
        episode_timesteps += 1

        if t < args.start_timesteps:
            action = env.action_space.sample()
        else:
            action = (
                policy.select_action(np.array(state), mu)
                + np.random.normal(0, max_action * args.expl_noise, size=action_dim)
            ).clip(-max_action, max_action)

        next_state, reward, terminated, truncated, _ = env.step(action)
        next_mu = mu
        done = terminated or truncated

        done_bool = float(done) if episode_timesteps < env.spec.max_episode_steps else 0
        replay_buffer.add(state, mu, action, next_state, next_mu, reward, done_bool)

        state = next_state
        episode_reward += reward

        if t >= args.start_timesteps:
            policy.train(replay_buffer, args.batch_size)

        if done:
            print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
            
            # 需要确保随机数生成不受seed影响
            # randomizer在上面已经创建了
            gravity_scale, friction_scale, mass_scale, gear_scale, mu = randomizer.sample()
            # 重新创建环境并重新用wrapper包装，这样来防止wrapper层层包裹
            # env.close()
            # env = gym.make(args.env)
            # env = utils.TargetDomainWrapper(
            #     env,
            #     gravity_scale=gravity_scale,
            #     friction_scale=friction_scale,
            #     mass_scale=mass_scale,
            #     gear_scale=gear_scale,
            # )
            # 重新设置环境参数, 调用在wrapper后env已经具有的update_params方法
            env.update_params(gravity_scale, friction_scale, mass_scale, gear_scale)

            state, _ = env.reset()
            episode_reward = 0
            episode_timesteps = 0
            episode_num += 1

        if (t + 1) % args.eval_freq == 0:
            evaluations.append(eval_policy(policy, args.env, args.seed))
            np.save(f"./results/{file_name}", evaluations)
            if args.save_model:
                policy.save(f"./models/{file_name}")
