# utils.py - Utility Classes and Functions for ACNR AI Control

import torch
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import torch.nn as nn
import numpy as np
import random
from collections import deque
from config import NUM_TRAJECTORIES, TRAJECTORY_LENGTH, ACTION_DIM, BATCH_SIZE_SURROGATE, LEARNING_RATE_SURROGATE, EPOCHS_SURROGATE, REPLAY_BUFFER_SIZE, device

class ReactorDataset(Dataset):
    def __init__(self, states, actions, next_states):
        self.states = torch.tensor(states, dtype=torch.float32)
        self.actions = torch.tensor(actions, dtype=torch.float32)
        self.next_states = torch.tensor(next_states, dtype=torch.float32)

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.next_states[idx]

def generate_dataset(simulator_class, num_trajectories=NUM_TRAJECTORIES, trajectory_length=TRAJECTORY_LENGTH):
    simulator = simulator_class()
    states = []
    actions = []
    next_states = []
    for _ in range(num_trajectories):
        simulator.reset()
        for t in range(trajectory_length):
            action = np.random.uniform(-1, 1, ACTION_DIM)  # Random actions for exploration
            state = simulator.state.copy()
            next_state, _, _ = simulator.step(action)
            states.append(state)
            actions.append(action)
            next_states.append(next_state)
    return np.array(states), np.array(actions), np.array(next_states)

def train_surrogate(model, dataset, epochs=EPOCHS_SURROGATE):
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE_SURROGATE, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_SURROGATE)
    criterion = nn.MSELoss()
    for epoch in range(epochs):
        total_loss = 0
        for states, actions, next_states in dataloader:
            states, actions, next_states = states.to(device), actions.to(device), next_states.to(device)
            preds = model(states, actions)
            loss = criterion(preds, next_states)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")
    return model

class ReplayBuffer:
    def __init__(self, capacity=REPLAY_BUFFER_SIZE):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*batch)
        return np.array(state), np.array(action), np.array(reward), np.array(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)
