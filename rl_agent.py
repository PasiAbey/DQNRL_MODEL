import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque

# --- DEEP Q-NETWORK ---
class DQN(nn.Module):
    def __init__(self, input_size=8, output_size=5):
        super(DQN, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_size)
        )
    def forward(self, x):
        return self.net(x)

# --- RL AGENT ---
class RLAgent:
    def __init__(self):
        self.input_size = 8
        self.output_size = 5
        self.model = DQN(self.input_size, self.output_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.gamma = 0.95
        self.memory = deque(maxlen=2000)

    def choose_action(self, state_vector):
        state_tensor = torch.FloatTensor(state_vector)
        with torch.no_grad():
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

# --- STATE VECTOR BUILDER ---
def get_rl_state_vector(level, duration_norm, risk_score, quiz_norm, consecutive_norm, daily_xp_norm):
    levels = ['Beginner', 'Intermediate', 'Expert']
    level_vec = [0, 0, 0]
    if level in levels:
        level_vec[levels.index(level)] = 1
    state_vector = np.array(level_vec + [duration_norm, risk_score, quiz_norm, consecutive_norm, daily_xp_norm])
    return state_vector