import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

# --- Parameters ---
GRID_SIZE = 10
NUM_EPISODES = 500
MAX_STEPS = 100
EPSILON_DECAY = 0.995
MIN_EPSILON = 0.01
GAMMA = 0.99
BATCH_SIZE = 64
MEMORY_SIZE = 10000
LEARNING_RATE = 0.001

# --- Neural Network ---
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.out(x)

# --- Environment (simplified placeholder) ---
def get_valid_actions(state):
    actions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # L R U D
    x, y = state
    valid = []
    for i, (dx, dy) in enumerate(actions):
        nx, ny = x + dx, y + dy
        if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
            valid.append(i)
    return valid

def take_action(state, action):
    dx, dy = [(0,-1), (0,1), (-1,0), (1,0)][action]
    nx, ny = state[0]+dx, state[1]+dy
    if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
        next_state = (nx, ny)
        reward = 1 if next_state == (9,9) else -0.01
        done = next_state == (9,9)
        return next_state, reward, done
    return state, -1, False  # Wall hit

def state_to_tensor(state):
    x = torch.zeros(GRID_SIZE * GRID_SIZE)
    x[state[0]*GRID_SIZE + state[1]] = 1.0
    return x.unsqueeze(0)

# --- Setup ---
q_net = QNetwork(GRID_SIZE * GRID_SIZE, 4)
optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
memory = deque(maxlen=MEMORY_SIZE)
epsilon = 1.0

# --- Training Loop ---
for episode in range(NUM_EPISODES):
    state = (0, 0)
    total_reward = 0
    for step in range(MAX_STEPS):
        state_tensor = state_to_tensor(state)

        # Epsilon-greedy action
        if random.random() < epsilon:
            action = random.choice(get_valid_actions(state))
        else:
            with torch.no_grad():
                q_vals = q_net(state_tensor)
                action = torch.argmax(q_vals).item()

        next_state, reward, done = take_action(state, action)
        memory.append((state, action, reward, next_state, done))
        state = next_state
        total_reward += reward

        # Learn from memory
        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            state_batch = torch.cat([state_to_tensor(s) for s in states])
            next_state_batch = torch.cat([state_to_tensor(s) for s in next_states])
            action_batch = torch.tensor(actions).unsqueeze(1)
            reward_batch = torch.tensor(rewards).float().unsqueeze(1)
            done_batch = torch.tensor(dones).float().unsqueeze(1)

            current_q = q_net(state_batch).gather(1, action_batch)
            with torch.no_grad():
                max_next_q = q_net(next_state_batch).max(1)[0].unsqueeze(1)
                target_q = reward_batch + GAMMA * max_next_q * (1 - done_batch)

            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if done:
            break

    epsilon = max(MIN_EPSILON, epsilon * EPSILON_DECAY)
    print(f"Episode {episode} | Total reward: {total_reward:.2f} | Epsilon: {epsilon:.3f}")

