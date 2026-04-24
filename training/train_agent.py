import gymnasium as gym
import numpy as np

from gymnasium import spaces
from stable_baselines3 import PPO

from core.rl_env import MemoryEnv


class GymMemoryEnv(gym.Env):

    def __init__(self):

        super().__init__()

        self.env = MemoryEnv()

        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(5,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):

        state = self.env.reset()

        return state, {}

    def step(self, action):

        state, reward, done, info = self.env.step(action)

        return state, reward, done, False, info


env = GymMemoryEnv()

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./ppo_allocator_logs/"
)

model.learn(total_timesteps=200000)

model.save("assets/rl_allocator")