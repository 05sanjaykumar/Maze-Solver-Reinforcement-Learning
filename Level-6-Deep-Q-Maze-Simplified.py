import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size,128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_size) 

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.rele(self.fc2(x))
        return self.fc3(x)
    
#initialise Q-network

state_size = 2 # For simplicity, use 2D state (e.g., (x, y) position in the maze)
action_size = 4 
q_network = QNetwork(state_size, action_size)

# Optimizer and loss function
optimizer = optim.Adam(q_network.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Sample state and action
state = torch.tensor([0, 0], dtype=torch.float32)  # Example state
action = torch.tensor([1], dtype=torch.int64)      # Example action (moving right)

# Forward pass: Get Q-values
q_values = q_network(state)
predicted_q_value = q_values[action]

# Assume target Q-value (from Bellman equation)
reward = 1  # Example reward for moving right
next_state = torch.tensor([0, 1], dtype=torch.float32)  # Next state after moving right
next_q_values = q_network(next_state)
max_next_q_value = torch.max(next_q_values).item()  # Max Q-value from next state

target_q_value = reward + 0.99 * max_next_q_value  # Bellman update

# Loss calculation
loss = loss_fn(predicted_q_value, torch.tensor([target_q_value], dtype=torch.float32))

# Backpropagation
optimizer.zero_grad()
loss.backward()
optimizer.step()

# Print result
print(f"Predicted Q-value: {predicted_q_value.item()}, Target Q-value: {target_q_value}")