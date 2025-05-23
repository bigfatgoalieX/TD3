import numpy as np
import torch
import gymnasium as gym
import mujoco


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
		gear_scale=1.0,
		world_mass_scale=1.0,
		torso_mass_scale=1.0,
		bthigh_mass_scale=1.0,
		bshin_mass_scale=1.0,
		bfoot_mass_scale=1.0,
		fthigh_mass_scale=1.0,
		fshin_mass_scale=1.0,
		ffoot_mass_scale=1.0,
		obs_noise_std=0.0, 
		action_noise_std=0.0
	):
		super().__init__(env)
		self.gravity_scale = gravity_scale
		self.friction_scale = friction_scale
		self.gear_scale = gear_scale
  
		self.world_mass_scale = world_mass_scale
		self.torso_mass_scale = torso_mass_scale
		self.bthigh_mass_scale = bthigh_mass_scale
		self.bshin_mass_scale = bshin_mass_scale
		self.bfoot_mass_scale = bfoot_mass_scale
		self.fthigh_mass_scale = fthigh_mass_scale
		self.fshin_mass_scale = fshin_mass_scale
		self.ffoot_mass_scale = ffoot_mass_scale
  
		self.obs_noise_std = obs_noise_std
		self.action_noise_std = action_noise_std
		self.modified = False
  
		# 记录原始的物理参数,防止叠加乘法偏出范围（1.2*1.2 = 1.44）
		self._original_gravity = self.env.unwrapped.model.opt.gravity.copy()
		self._original_friction = self.env.unwrapped.model.geom_friction.copy()
		self._original_gear = self.env.unwrapped.model.actuator_gear.copy()
		self._original_mass = self.env.unwrapped.model.body_mass.copy()



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
		# 注意参数都是在原始值的基础上进行缩放，而不是每次叠加乘法
		model.opt.gravity[2] = self._original_gravity[2] * self.gravity_scale 
		model.geom_friction[:, 0] = self._original_friction[:, 0] * self.friction_scale
		model.actuator_gear[:] = self._original_gear * self.gear_scale
		# model.body_mass[:] = self._original_mass * self.mass_scale
		for i in range(model.nbody):
			name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
			if name == "torso":
				model.body_mass[i] = self._original_mass[i] * self.torso_mass_scale
			elif name == "bthigh":
				model.body_mass[i] = self._original_mass[i] * self.bthigh_mass_scale
			elif name == "bshin":
				model.body_mass[i] = self._original_mass[i] * self.bshin_mass_scale
			elif name == "bfoot":
				model.body_mass[i] = self._original_mass[i] * self.bfoot_mass_scale
			elif name == "fthigh":
				model.body_mass[i] = self._original_mass[i] * self.fthigh_mass_scale
			elif name == "fshin":
				model.body_mass[i] = self._original_mass[i] * self.fshin_mass_scale
			elif name == "ffoot":
				model.body_mass[i] = self._original_mass[i] * self.ffoot_mass_scale
  
	
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
	
	def update_params(self, 
                   gravity_scale=None, 
                   friction_scale=None, 
                   gear_scale=None,
				   torso_mass_scale=None,
				   bthigh_mass_scale=None,
				   bshin_mass_scale=None,
				   bfoot_mass_scale=None,
				   fthigh_mass_scale=None,
				   fshin_mass_scale=None,
				   ffoot_mass_scale=None): 
		self.gravity_scale = gravity_scale if gravity_scale is not None else self.gravity_scale
		self.friction_scale = friction_scale if friction_scale is not None else self.friction_scale
		self.gear_scale = gear_scale if gear_scale is not None else self.gear_scale
		self.torso_mass_scale = torso_mass_scale if torso_mass_scale is not None else self.torso_mass_scale
		self.bthigh_mass_scale = bthigh_mass_scale if bthigh_mass_scale is not None else self.bthigh_mass_scale
		self.bshin_mass_scale = bshin_mass_scale if bshin_mass_scale is not None else self.bshin_mass_scale
		self.bfoot_mass_scale = bfoot_mass_scale if bfoot_mass_scale is not None else self.bfoot_mass_scale
		self.fthigh_mass_scale = fthigh_mass_scale if fthigh_mass_scale is not None else self.fthigh_mass_scale
		self.fshin_mass_scale = fshin_mass_scale if fshin_mass_scale is not None else self.fshin_mass_scale
		self.ffoot_mass_scale = ffoot_mass_scale if ffoot_mass_scale is not None else self.ffoot_mass_scale
		self.modified = False

class PhysicsRandomizer:
	def __init__(self, param_ranges):
		"""
		param_ranges: dict, e.g.
		{
			"gravity_scale": [0.5, 1.5],
			"friction_scale": [0.5, 2.5],
			"gear_scale": [0.8, 1.2],
			"torso_mass_scale": [0.5, 1.5],	
			"bthigh_mass_scale": [0.5, 1.5],
			"bshin_mass_scale": [0.5, 1.5],
			"bfoot_mass_scale": [0.5, 1.5],
			"fthigh_mass_scale": [0.5, 1.5],
			"fshin_mass_scale": [0.5, 1.5],
			"ffoot_mass_scale": [0.5, 1.5],
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