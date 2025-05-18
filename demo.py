import gymnasium as gym
import mujoco
env = gym.make("HalfCheetah-v5")
env.reset()

for i in range(env.unwrapped.model.nbody):
    name = mujoco.mj_id2name(env.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, i)
    mass = env.unwrapped.model.body_mass[i]
    print(f"{i}: {name} -> mass: {mass:.3f}")

