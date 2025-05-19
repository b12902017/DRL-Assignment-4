import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

# Do not modify the input of the 'act' function and the '__init__' function. 


LOG_STD_MIN = -20
LOG_STD_MAX = 2
class ActorNet(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim, device="cpu"):
        super(ActorNet, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )

        self.mean_linear = nn.Linear(hidden_dim, action_dim)
        self.log_std_linear = nn.Linear(hidden_dim, action_dim)
        self.action_scale = torch.tensor(2.0, dtype=torch.float32, device=device)

        self.to(device)

    def forward(self, state):
        x = self.feature(state)
        mean = self.mean_linear(x)
        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, min=LOG_STD_MIN, max=LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = torch.exp(log_std)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale

        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)

        mean = torch.tanh(mean) * self.action_scale
        return action, log_prob, mean

class Agent(object):
    """Agent that acts randomly."""
    def __init__(self):
        # Pendulum-v1 has a Box action space with shape (1,)
        # Actions are in the range [-2.0, 2.0]
        self.action_space = gym.spaces.Box(-2.0, 2.0, (1,), np.float32)
        self.policy = ActorNet(3, 1, 64, device="cpu")

        pth = "Q1_episode1000.pt"
        checkpoint = torch.load(pth, map_location="cpu")
        self.policy.load_state_dict(checkpoint['policy_state_dict'])

    def act(self, observation):
        state = torch.from_numpy(observation).unsqueeze(0).float()
        _, _, action = self.policy.sample(state)
        return action.detach().numpy()[0]