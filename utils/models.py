import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class CustomLSTM(BaseFeaturesExtractor):
    def __init__(self, observation_space, feature_dim=64):
        super().__init__(observation_space, features_dim=feature_dim)
        input_dim = observation_space.shape[0]
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, feature_dim)

    def forward(self, observations):
        # observations: (batch, features)
        # Add time dimension for LSTM: (batch, seq_len=1, features)
        x = observations.unsqueeze(1)
        lstm_out, _ = self.lstm(x)
        # Take the output of the last time step
        x = lstm_out[:, -1, :]
        x = self.fc(x)
        return x
