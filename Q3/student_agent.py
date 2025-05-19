import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn

# Do not modify the input of the 'act' function and the '__init__' function. 



class GaussianPolicy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_dim, 256), nn.ReLU(),
                                 nn.Linear(256, 256), nn.ReLU())
        self.mean_linear = nn.Linear(256, act_dim)
        self.log_std_linear = nn.Linear(256, act_dim)
        self.LOG_STD_MIN = -20
        self.LOG_STD_MAX = 2

    def forward(self, state):
        x = self.net(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        std = log_std.exp()
        return mean, std
    

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        self.action_space = gym.spaces.Box(-1.0, 1.0, (21,), np.float64)
        self.policy = GaussianPolicy(67, 21)
        self.policy.load_state_dict(torch.load("GPT_sac_episode_1550.pth", map_location=torch.device('cpu'))['policy_state_dict'])

    def act(self, observation):
        x = torch.from_numpy(observation).unsqueeze(0).float()
        with torch.no_grad():
            action, _ = self.policy(x)
        return action.cpu().numpy()[0]
