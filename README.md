# ACNR AI Control System

This repository implements an AI-driven control system for the Autocatalytic Carbon-Negative Reactor (ACNR) using a CNN-LSTM surrogate model and Soft Actor-Critic (SAC) reinforcement learning.

## Files
- `config.py`: Constants and hyperparameters.
- `models.py`: Neural network definitions.
- `utils.py`: Datasets, training functions, and replay buffer.
- `agent.py`: SAC agent and digital twin environment.
- `main.py`: Training and deployment scripts.

## Setup
1. Install dependencies: `pip install -r requirements.txt`
2. Run: `python main.py`

Note: This is a simulation; replace dummy simulator with real physics engine for production.
