
import numpy as np
import gymnasium as gym
from dm_control import suite
from gymnasium.wrappers import FlattenObservation
from shimmy import DmControlCompatibilityV0 as DmControltoGymnasium
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.distributions import Normal
from collections import deque
import random
from torchrl.data import ReplayBuffer, TensorDictReplayBuffer, LazyTensorStorage
from torchrl.data import SamplerWithoutReplacement
import sys

# Environment Setup
def make_env():
    env = suite.load("humanoid", "walk")
    env = DmControltoGymnasium(env, render_mode="rgb_array", render_kwargs={"width": 256, "height": 256})
    env = FlattenObservation(env)
    return env

env = make_env()
obs_dim = env.observation_space.shape[0]
act_dim = env.action_space.shape[0]
act_limit = 1.0
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Replay Buffer
class RB:
    def __init__(self, capacity=1000000, device="cpu"):
        self.device = torch.device(device)
        self.buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=capacity, device=self.device),
            sampler=SamplerWithoutReplacement()
        )

    def add(self, state, action, reward, next_state, done):
        transition = {
            "state": torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0),
            "action": torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0),
            "reward": torch.tensor(reward, dtype=torch.float32, device=self.device).unsqueeze(0),
            "next_state": torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0),
            "done": torch.tensor(done, device=self.device).unsqueeze(0)
        }
        self.buffer.add(transition)

    def sample(self, batch_size):
        batch = self.buffer.sample(batch_size)
        return (
            batch["state"],
            batch["action"],
            batch["reward"],
            batch["next_state"],
            batch["done"]
        )

    def __len__(self):
        return len(self.buffer)

# Networks
def _init_weight(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
class MLP(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, out_dim)
        )
        self.apply(_init_weight)
    def forward(self, x):
        return self.net(x)

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

    def sample(self, state):
        mean, std = self.forward(state)
        normal = Normal(mean, std)
        x_t = normal.rsample()
        y_t = torch.tanh(x_t)
        action = y_t * act_limit

        log_prob = normal.log_prob(x_t) - torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

policy = GaussianPolicy(obs_dim, act_dim).to(device)
q1 = MLP(obs_dim + act_dim, 1).to(device)
q2 = MLP(obs_dim + act_dim, 1).to(device)
q1_target = MLP(obs_dim + act_dim, 1).to(device)
q2_target = MLP(obs_dim + act_dim, 1).to(device)
q1_target.load_state_dict(q1.state_dict())
q2_target.load_state_dict(q2.state_dict())

policy_optim = Adam(policy.parameters(), lr=3e-4)
q1_optim = Adam(q1.parameters(), lr=3e-4)
q2_optim = Adam(q2.parameters(), lr=3e-4)

# Alpha tuning setup
target_entropy = -act_dim
log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha_optim = Adam([log_alpha], lr=1e-4)

replay_buffer = RB(device=device)

def update(batch_size=256, gamma=0.99, tau=0.005):
    if len(replay_buffer) < 10000:
        return

    state, action, reward, next_state, done = replay_buffer.sample(batch_size)
    state = state.squeeze(1)
    action = action.squeeze(1)
    next_state = next_state.squeeze(1)

    with torch.no_grad():
        next_action, next_log_prob = policy.sample(next_state)
        q1_next = q1_target(torch.cat([next_state, next_action], dim=-1))
        q2_next = q2_target(torch.cat([next_state, next_action], dim=-1))
        
        alpha = log_alpha.exp()
        q_target = reward + gamma * (1 - done) * (torch.min(q1_next, q2_next) - alpha * next_log_prob)

    q1_loss = F.mse_loss(q1(torch.cat([state, action], dim=-1)), q_target)
    q2_loss = F.mse_loss(q2(torch.cat([state, action], dim=-1)), q_target)

    q1_optim.zero_grad()
    q1_loss.backward()
    q1_optim.step()

    q2_optim.zero_grad()
    q2_loss.backward()
    q2_optim.step()

    new_action, log_prob = policy.sample(state)
    q1_pi = q1(torch.cat([state, new_action], dim=-1))
    q2_pi = q2(torch.cat([state, new_action], dim=-1))
    min_q = torch.min(q1_pi, q2_pi)
    alpha = log_alpha.exp()
    policy_loss = (alpha * log_prob - min_q).mean()

    policy_optim.zero_grad()
    policy_loss.backward()
    policy_optim.step()

    # Update alpha
    alpha_loss = -(log_alpha * (log_prob + target_entropy).detach()).mean()
    alpha_optim.zero_grad()
    alpha_loss.backward()
    alpha_optim.step()

    for param, target_param in zip(q1.parameters(), q1_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
    for param, target_param in zip(q2.parameters(), q2_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    return q1_loss.item(), q2_loss.item(), policy_loss.item(), alpha_loss.item(), alpha.item()

episodes = 50000
epsilon = 1
for episode in range(episodes):
    state, _ = env.reset()
    total_reward = 0
    for t in range(1000):
        state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            if episode < 50:
                action = env.action_space.sample()
            else:
                action, _ = policy.sample(state_tensor)
                action = action.cpu().numpy()[0]

        if random.random() < epsilon:
            action += np.random.normal(0, 0.2, size=action.shape)
            action = np.clip(action, -1, 1)
        
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated
        replay_buffer.add(state, action, reward, next_state, float(done))
        state = next_state
        total_reward += reward
        result = update()
        if done:
            break
        
    if epsilon > 0.05:
        epsilon *= 0.999
    
    if result:
        q1_loss, q2_loss, pol_loss, ent_loss, alpha = result
        print(f"Ep {episode+1} | R: {total_reward:.1f} | Q1: {q1_loss:.2f}, Q2: {q2_loss:.2f}, Pi: {pol_loss:.2f}, α: {alpha:.4f}")
    else:
        print(f"Ep {episode+1} | R: {total_reward:.1f}")

    if (episode+1) % 50 == 0 and total_reward > 300:
        print(f"Saving model at episode {episode+1}")
        torch.save({'policy_state_dict': policy.state_dict(),
                  'q1_state_dict': q1.state_dict(),
                  'q2_state_dict': q2.state_dict()}, f"/tmp2/b12902017/drl-Q4/GPT_sac_episode_{episode+1}.pth")

    sys.stdout.flush()

torch.save({'policy_state_dict': policy.state_dict(),
                  'q1_state_dict': q1.state_dict(),
                  'q2_state_dict': q2.state_dict()}, f"/tmp2/b12902017/drl-Q4/GPT_sac_checkpoint.pth")
