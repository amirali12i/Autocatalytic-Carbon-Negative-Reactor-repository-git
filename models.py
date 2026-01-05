# models.py - Neural Network Models for ACNR AI Control

import torch
import torch.nn as nn
from config import STATE_DIM, ACTION_DIM, CNN_FILTERS, LSTM_HIDDEN, LSTM_LAYERS, device

class SurrogateModel(nn.Module):
    def __init__(self):
        super(SurrogateModel, self).__init__()
        # CNN for spatial features (treat state as 1D sequence)
        self.cnn = nn.Sequential(
            nn.Conv1d(1, CNN_FILTERS[0], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(CNN_FILTERS[0], CNN_FILTERS[1], kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        cnn_out_size = CNN_FILTERS[1] * STATE_DIM  # Approximate
        # LSTM for temporal dynamics
        self.lstm = nn.LSTM(cnn_out_size + ACTION_DIM, LSTM_HIDDEN, LSTM_LAYERS, batch_first=True)
        # Output layer
        self.fc = nn.Linear(LSTM_HIDDEN, STATE_DIM)

    def forward(self, state, action):
        # State: batch x state_dim, reshape to batch x 1 x state_dim for CNN
        state = state.unsqueeze(1)  # Add channel dim
        spatial = self.cnn(state)
        combined = torch.cat([spatial, action], dim=1).unsqueeze(1)  # Batch x 1 x (spatial + action)
        lstm_out, _ = self.lstm(combined)
        next_state = self.fc(lstm_out.squeeze(1))
        return next_state

class Actor(nn.Module):
    def __init__(self, max_action=1.0):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, ACTION_DIM)
        self.log_std = nn.Linear(256, ACTION_DIM)
        self.max_action = max_action

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        mu = self.mu(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mu, std

    def sample(self, state):
        mu, std = self.forward(state)
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()
        action = torch.tanh(x) * self.max_action
        log_prob = normal.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        return action, log_prob

class Critic(nn.Module):
    def __init__(self):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(STATE_DIM + ACTION_DIM, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q = nn.Linear(256, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.q(x)
