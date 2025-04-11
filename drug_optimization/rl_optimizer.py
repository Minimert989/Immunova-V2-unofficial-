# drug_optimization/rl_optimizer.py

import numpy as np
import torch
from collections import deque

class DrugRLAgent:
    def __init__(self, model, lr=0.001, gamma=0.99):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.gamma = gamma
        self.memory = deque(maxlen=10000)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        with torch.no_grad():
            return torch.argmax(self.model(state)).item()

    def learn(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = np.random.choice(len(self.memory), batch_size, replace=False)
        for idx in batch:
            state, action, reward, next_state, done = self.memory[idx]
            target = reward
            if not done:
                target += self.gamma * torch.max(self.model(next_state)).item()

            output = self.model(state)[action]
            loss = F.mse_loss(output, torch.tensor(target))

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
