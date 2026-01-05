# agent.py - SAC Agent and Digital Twin Environment for ACNR AI Control

import torch
import torch.optim as optim
import torch.nn as nn
import copy
import numpy as np
from config import STATE_DIM, ACTION_DIM, GAMMA, TAU, LEARNING_RATE_SAC, TARGET_ENTROPY, device
from models import Actor, Critic

class SACAgent:
    def __init__(self):
        self.actor = Actor().to(device)
        self.critic1 = Critic().to(device)
        self.critic2 = Critic().to(device)
        self.target_critic1 = copy.deepcopy(self.critic1)
        self.target_critic2 = copy.deepcopy(self.critic2)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LEARNING_RATE_SAC)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LEARNING_RATE_SAC)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LEARNING_RATE_SAC)
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=device)  # Initial alpha=0.2
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LEARNING_RATE_SAC)
        self.target_entropy = TARGET_ENTROPY

    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        action, _ = self.actor.sample(state)
        return action.cpu().detach().numpy().flatten()

    def update(self, replay_buffer, batch_size):
        if len(replay_buffer) < batch_size:
            return
        state, action, reward, next_state, done = replay_buffer.sample(batch_size)
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = torch.tensor(action, dtype=torch.float32).to(device)
        reward = torch.tensor(reward, dtype=torch.float32).unsqueeze(1).to(device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(device)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(device)

        # Critic update
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_q1 = self.target_critic1(next_state, next_action)
            target_q2 = self.target_critic2(next_state, next_action)
            target_q = torch.min(target_q1, target_q2) - self.log_alpha.exp() * next_log_prob
            target_q = reward + (1 - done) * GAMMA * target_q

        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        critic1_loss = nn.MSELoss()(q1, target_q)
        critic2_loss = nn.MSELoss()(q2, target_q)

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Actor update
        pi_action, log_prob = self.actor.sample(state)
        q1_pi = self.critic1(state, pi_action)
        q2_pi = self.critic2(state, pi_action)
        min_q_pi = torch.min(q1_pi, q2_pi)
        actor_loss = (self.log_alpha.exp() * log_prob - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Alpha update
        alpha_loss = -(self.log_alpha.exp() * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # Soft update targets
        for param, target_param in zip(self.critic1.parameters(), self.target_critic1.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)
        for param, target_param in zip(self.critic2.parameters(), self.target_critic2.parameters()):
            target_param.data.copy_(TAU * param.data + (1 - TAU) * target_param.data)

class DigitalTwinEnv:
    def __init__(self, surrogate_model, simulator_class):
        self.surrogate = surrogate_model
        self.simulator = simulator_class()  # For initial reset, but use surrogate for steps
        self.reset()

    def reset(self):
        self.simulator.reset()
        self.state = torch.tensor(self.simulator.state, dtype=torch.float32).to(device)
        return self.state.cpu().numpy()

    def step(self, action):
        action_tensor = torch.tensor(action, dtype=torch.float32).unsqueeze(0).to(device)
        state_tensor = self.state.unsqueeze(0)
        next_state = self.surrogate(state_tensor, action_tensor).squeeze(0)
        self.state = next_state

        # Compute reward (dummy: H2 yield - penalties)
        h2_mean = next_state[-1 - NUM_SPECIES * REACTOR_LENGTH : -1].reshape(REACTOR_LENGTH, NUM_SPECIES)[:, 2].mean()
        h2s_mean = next_state[-1 - NUM_SPECIES * REACTOR_LENGTH : -1].reshape(REACTOR_LENGTH, NUM_SPECIES)[:, 4].mean()
        temp = next_state[-1].item()
        reward = h2_mean - 0.1 * h2s_mean - 0.01 * abs(temp - 850)  # Target temp 850K
        done = temp > 1100 or temp < 500
        return next_state.cpu().numpy(), reward, done
