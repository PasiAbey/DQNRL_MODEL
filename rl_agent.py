import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import os

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
        
        # Epsilon variables needed for the replay function and saving state
        self.epsilon = 0.01
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        
        self.memory = deque(maxlen=2000)

    # Function to safely load your Colab .pth file on a cloud CPU
    def load_pretrained_model(self, filepath):
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=torch.device('cpu'), weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            # Safely load epsilon if it exists in the checkpoint
            self.epsilon = checkpoint.get('epsilon', 0.01)
            print(f"✅ Successfully loaded pre-trained model from {filepath}")
        else:
            print(f"⚠️ Warning: Model file {filepath} not found. Using untrained, random weights.")

    def choose_action(self, state_vector):
        state_tensor = torch.FloatTensor(state_vector)
        # In production, we usually stick to greedy actions (no random exploration) 
        # to ensure the user gets the best possible prediction.
        with torch.no_grad():
            q_values = self.model(state_tensor)
            action = torch.argmax(q_values).item()
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # NEW: The backpropagation engine used by train.py
    def replay(self, batch_size=32):
        if len(self.memory) < batch_size:
            return False

        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            state_t = torch.FloatTensor(state)
            next_state_t = torch.FloatTensor(next_state)
            target = reward

            # Bellman Equation
            if not done:
                target = reward + self.gamma * torch.max(self.model(next_state_t)).item()

            target_f = self.model(state_t).clone()
            target_f[action] = target

            # Gradient Descent
            self.optimizer.zero_grad()
            loss = self.loss_fn(self.model(state_t), target_f)
            loss.backward()
            self.optimizer.step()

        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return True

# --- STATE VECTOR BUILDER ---
def get_rl_state_vector(level, duration_norm, risk_score, quiz_norm, consecutive_norm, daily_xp_norm):
    levels = ['Beginner', 'Intermediate', 'Expert']
    level_vec = [0, 0, 0]
    if level in levels:
        level_vec[levels.index(level)] = 1
    state_vector = np.array(level_vec + [duration_norm, risk_score, quiz_norm, consecutive_norm, daily_xp_norm])
    return state_vector