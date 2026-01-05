# config.py - Constants and Hyperparameters for ACNR AI Control

import torch

# Reactor simulation parameters (dummy values for illustration)
NUM_SPECIES = 5  # e.g., CH4, CO2, H2, CO, H2S
REACTOR_LENGTH = 100  # Discretized axial points
STATE_DIM = NUM_SPECIES * REACTOR_LENGTH + 1  # Concentrations + temperature
ACTION_DIM = 2  # e.g., slipstream ratio, O2 flow
MAX_EPISODE_STEPS = 1000
NUM_TRAJECTORIES = 500000
TRAJECTORY_LENGTH = 1000

# Surrogate Model Hyperparameters
CNN_FILTERS = [64, 128]
LSTM_HIDDEN = 256
LSTM_LAYERS = 2
LEARNING_RATE_SURROGATE = 1e-3
BATCH_SIZE_SURROGATE = 256
EPOCHS_SURROGATE = 50

# SAC Hyperparameters
GAMMA = 0.99  # Discount factor
TAU = 0.005  # Soft update coefficient
ALPHA = 0.2  # Entropy regularization
LEARNING_RATE_SAC = 3e-4
REPLAY_BUFFER_SIZE = int(1e6)
BATCH_SIZE_SAC = 256
NUM_EPISODES_SAC = int(1e6)
TARGET_ENTROPY = -ACTION_DIM  # For automatic alpha tuning

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
