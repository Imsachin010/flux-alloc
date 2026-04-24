from sb3_contrib import MaskablePPO
import gymnasium as gym
import numpy as np
from gymnasium import spaces

from core.rl_env_direct_final import DirectPlacementEnv
from policy.custom_policy_transformer import TransformerFeatureExtractor


class MaskedEnv(gym.Env):

    def __init__(self):
        self.env = DirectPlacementEnv()

        self.action_space = spaces.Discrete(20)

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(8 + 20*5 + 20,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        state = self.env.reset()
        return state, {}

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        return state, reward, done, False, info

    def action_masks(self):
        return self.env.get_action_mask()


env = MaskedEnv()

policy_kwargs = dict(
    features_extractor_class=TransformerFeatureExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

model = MaskablePPO(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,
    verbose=1,
    device="cuda",
    n_steps=2048,
    batch_size=256,
)

model.learn(total_timesteps=700000)

model.save("assets/rl_direct_allocator")