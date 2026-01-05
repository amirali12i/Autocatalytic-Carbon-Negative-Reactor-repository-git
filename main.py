# main.py - Main Execution for ACNR AI Control

import numpy as np
from config import NUM_TRAJECTORIES, TRAJECTORY_LENGTH, NUM_EPISODES_SAC, MAX_EPISODE_STEPS, BATCH_SIZE_SAC
from models import SurrogateModel
from utils import generate_dataset, ReactorDataset, train_surrogate, ReplayBuffer
from agent import SACAgent, DigitalTwinEnv

class HighFidelitySimulator:
    def __init__(self):
        from config import REACTOR_LENGTH, NUM_SPECIES
        self.dt = 0.01  # Time step
        self.length = REACTOR_LENGTH
        self.species = NUM_SPECIES
        self.reset()

    def reset(self):
        from config import REACTOR_LENGTH, NUM_SPECIES
        self.concentrations = np.random.uniform(0.4, 0.6, size=(self.length, self.species))  # Axial profiles
        self.temperature = np.random.uniform(700, 900)  # Kelvin
        self.state = np.concatenate([self.concentrations.flatten(), [self.temperature]])

    def step(self, action):
        from config import REACTOR_LENGTH, NUM_SPECIES
        # Action: [slipstream_ratio, o2_flow]
        slipstream = action[0] * 0.2  # Scale to 0-20%
        o2 = action[1] * 10  # Scale

        # Simulate reactions (dummy kinetics)
        for i in range(self.length):
            # MDR: CH4 + CO2 -> 2CO + 2H2
            rate_mdr = 0.1 * self.concentrations[i, 0] * self.concentrations[i, 1]
            self.concentrations[i, 0] -= rate_mdr * self.dt
            self.concentrations[i, 1] -= rate_mdr * self.dt
            self.concentrations[i, 2] += 2 * rate_mdr * self.dt  # H2
            self.concentrations[i, 3] += 2 * rate_mdr * self.dt  # CO

            # CPOX influence from slipstream
            rate_cpox = slipstream * o2 * self.concentrations[i, 0] * 0.05
            self.concentrations[i, 0] -= rate_cpox * self.dt
            self.concentrations[i, 3] += rate_cpox * self.dt  # CO
            self.concentrations[i, 2] += 2 * rate_cpox * self.dt  # H2

            # H2S purification (dummy)
            self.concentrations[i, 4] *= 0.99  # Decay

        # Energy balance (heat from CPOX)
        heat_gen = rate_cpox * 36  # Exothermic
        self.temperature += (heat_gen - 247 * rate_mdr) * self.dt / 1000  # Simplified

        # Clip values
        self.concentrations = np.clip(self.concentrations, 0, 1)
        self.temperature = np.clip(self.temperature, 500, 1200)

        self.state = np.concatenate([self.concentrations.flatten(), [self.temperature]])
        reward = self.concentrations[:, 2].mean() - 0.1 * self.concentrations[:, 4].mean()  # H2 yield - H2S penalty
        done = self.temperature > 1100 or np.any(self.concentrations < 0)  # Safety bounds
        return self.state, reward, done

def offline_training():
    # Step 1: Generate dataset
    print("Generating dataset...")
    states, actions, next_states = generate_dataset(HighFidelitySimulator, num_trajectories=NUM_TRAJECTORIES // 10, trajectory_length=TRAJECTORY_LENGTH // 10)  # Reduce for demo
    dataset = ReactorDataset(states, actions, next_states)

    # Step 2: Train surrogate
    print("Training surrogate model...")
    surrogate = SurrogateModel().to(device)
    surrogate = train_surrogate(surrogate, dataset)

    # Step 3: Create digital twin environment
    env = DigitalTwinEnv(surrogate, HighFidelitySimulator)

    # Step 4: Train SAC
    print("Training SAC agent...")
    agent = SACAgent()
    replay_buffer = ReplayBuffer()

    for episode in range(NUM_EPISODES_SAC // 100):  # Reduce for demo
        state = env.reset()
        episode_reward = 0
        for t in range(MAX_EPISODE_STEPS):
            action = agent.select_action(state)
            next_state, reward, done = env.step(action)
            replay_buffer.push(state, action, reward, next_state, done)
            agent.update(replay_buffer, BATCH_SIZE_SAC)
            state = next_state
            episode_reward += reward
            if done:
                break
        print(f"Episode {episode+1}, Reward: {episode_reward}")

    return agent, surrogate

def online_deployment(agent, num_steps=1000):
    simulator = HighFidelitySimulator()  # Real physical interface in practice
    state = simulator.state
    total_reward = 0
    for t in range(num_steps):
        action = agent.select_action(state)
        next_state, reward, done = simulator.step(action)  # Real step
        total_reward += reward
        state = next_state
        if done:
            print("Safety bound reached.")
            break
    print(f"Total reward: {total_reward}")

if __name__ == "__main__":
    agent, surrogate = offline_training()
    online_deployment(agent)
