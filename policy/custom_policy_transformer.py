import torch as th
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import gymnasium as gym

class TransformerFeatureExtractor(BaseFeaturesExtractor):

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 128):
        super().__init__(observation_space, features_dim)

        self.max_blocks = 20
        self.d_model = 64

        self.global_net = nn.Linear(8, 64)  # 5 globals + 3 lookahead

        self.block_proj = nn.Linear(5, self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

    def forward(self, obs):

        batch = obs.shape[0]

        global_feats = obs[:, :8]
        block_feats = obs[:, 8:-self.max_blocks].view(batch, self.max_blocks, 5)
        # Note: action masks are extracted and used directly by MaskablePPO internally.

        g = self.global_net(global_feats)
        b = self.block_proj(block_feats)
        b = self.transformer(b)

        pooled = b.mean(dim=1)

        return th.cat([pooled, g], dim=1)