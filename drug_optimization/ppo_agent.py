# ppo_agent.py
# Proximal Policy Optimization (PPO) agent for personalized treatment sequence learning.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        logits = self.fc(state)
        return F.softmax(logits, dim=-1)

class ValueNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state):
        return self.fc(state)

class PPOAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, eps_clip=0.2):
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.value = ValueNetwork(state_dim)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.value_optimizer = torch.optim.Adam(self.value.parameters(), lr=lr)
        self.gamma = gamma
        self.eps_clip = eps_clip

    def select_action(self, state):
        probs = self.policy(state)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), dist.log_prob(action), dist.entropy()

    def compute_returns(self, rewards, dones):
        R = 0
        returns = []
        for r, done in zip(reversed(rewards), reversed(dones)):
            if done:
                R = 0
            R = r + self.gamma * R
            returns.insert(0, R)
        return torch.tensor(returns, dtype=torch.float32)

    def update(self, states, actions, log_probs_old, returns):
        values = self.value(states).squeeze()
        advantages = returns - values.detach()

        # PPO update for policy
        new_probs = self.policy(states)
        dist = Categorical(new_probs)
        log_probs = dist.log_prob(actions)
        ratio = torch.exp(log_probs - log_probs_old)

        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
        loss_pi = -torch.min(surr1, surr2).mean()

        self.policy_optimizer.zero_grad()
        loss_pi.backward()
        self.policy_optimizer.step()

        # Update value network
        loss_v = F.mse_loss(self.value(states).squeeze(), returns)
        self.value_optimizer.zero_grad()
        loss_v.backward()
        self.value_optimizer.step()
