import numpy as np
import torch
import gymnasium as gym


class ReplayBuffer(object):
	def __init__(self, state_dim, mu_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.mu = np.zeros((max_size, mu_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.next_mu = np.zeros((max_size, mu_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, mu, action, next_state, next_mu, reward, done):
		self.state[self.ptr] = state
		self.mu[self.ptr] = mu
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.next_mu[self.ptr] = next_mu
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.mu[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.next_mu[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device)
		)
  
# 修改环境wrapper
class TargetDomainWrapper(gym.Wrapper):
	def __init__(self, env, 
		gravity_scale=1.0, 
		friction_scale=1.0, 
		mass_scale=1.0,
		obs_noise_std=0.0, 
		action_noise_std=0.0
	):
		super().__init__(env)
		self.gravity_scale = gravity_scale
		self.friction_scale = friction_scale
		self.mass_scale = mass_scale
		self.obs_noise_std = obs_noise_std
		self.action_noise_std = action_noise_std
		self.modified = False

	def reset(self, **kwargs):
		if not self.modified:
			self.modify_physics()
			self.modified = True
		result = self.env.reset(**kwargs)
		# 兼容gymnasium的新格式
		if isinstance(result, tuple):
			obs, info = result
			obs = self.add_obs_noise(obs)
			return obs, info
		# 旧版本gym格式
		else:
			obs = self.add_obs_noise(result)
			return obs
		
	def step(self, action):
		# 添加动作噪声
		noisy_action = self.add_action_noise(action)
		obs, reward, terminated, truncated, info = self.env.step(noisy_action)
		# 添加观测噪声
		return self.add_obs_noise(obs), reward, terminated, truncated, info

	def modify_physics(self):
		model = self.env.unwrapped.model
		model.opt.gravity[2] *= self.gravity_scale  # 重力
		model.geom_friction[:, 0] *= self.friction_scale #摩擦
		model.body_mass[:] *= self.mass_scale  # 质量
	
	def add_obs_noise(self, obs):
		# 兼容gymnasium的新格式
		if isinstance(obs, tuple):
			obs = obs[0]
		if self.obs_noise_std > 0.0:
			noise = np.random.normal(0, self.obs_noise_std, size=obs.shape)
			return obs + noise
		return obs
	
	def add_action_noise(self, action):
		# 兼容gymnasium的新格式
		if isinstance(action, tuple):
			action = action[0]
		if self.action_noise_std > 0.0:
			noise = np.random.normal(0, self.action_noise_std, size=action.shape)
			noisy_action = action + noise
			# clip action to valid range
			return np.clip(noisy_action, self.action_space.low, self.action_space.high)
		return action

class PhysicsRandomizer:
    def __init__(self, param_ranges):
        """
        param_ranges: dict, e.g.
        {
            "gravity": [-15.0, -5.0],
            "mass": [0.5, 2.0],
            "friction": [0.1, 1.0],
        }
        """
        self.param_ranges = param_ranges
        self.param_names = list(param_ranges.keys())
        self.rng = np.random.default_rng() # 不受seed影响的随机数生成器

    def sample(self):
        """
        Returns:
        - each param in order: gravity_val, mass_val, friction_val, ...
        - mu: concatenated vector (np.array)
        """
        sampled_values = []
        for name in self.param_names:
            low, high = self.param_ranges[name]
            val = self.rng.uniform(low, high)
            sampled_values.append(val)
        
        mu = np.array(sampled_values, dtype=np.float32)
        return (*sampled_values, mu)