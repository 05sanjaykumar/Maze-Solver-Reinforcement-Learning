#Base using Bellmans equation to solve our maze

import numpy as np
import random

ACTIONS = {
    0: (-1, 0),  # left
    1: (1, 0),   # right
    2: (0, -1),  # up
    3: (0, 1)    # down
}

class MazeEnv:
    def __init__(self,grid):
        self.grid = grid
        self.start = (0,0)
        self.goal = (9,9)
        self.state = self.start
    
    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        
        x,y = self.state
        dx,dy = ACTIONS[action]
        new_x,new_y = x+dx,y+dy

        if 0<=new_x<10 and 0<=new_y<10 and self.grid[new_x][new_y] != '█':
            self.state = (new_x,new_y)

        else:
            return self.state, -1, False # giving penality for mistakes

        if self.state == self.goal:
            return self.state, 10, True 
        
        return self.state, -0.1, False # Small penalty for each move (to encourage shorter paths)

env = MazeEnv([
    ['S', '.', '.', '.', '█', '.', '.', '.', '.', '.'],
    ['█', '█', '.', '█', '.', '█', '█', '█', '.', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '█', '.', '.'],
    ['.', '█', '█', '█', '█', '.', '█', '█', '█', '.'],
    ['.', '.', '.', '.', '█', '.', '.', '.', '.', '.'],
    ['█', '█', '█', '.', '█', '█', '█', '.', '█', '.'],
    ['.', '.', '.', '.', '.', '.', '.', '.', '.', '.'],
    ['█', '█', '█', '█', '.', '█', '█', '█', '█', '█'],
    ['.', '.', '.', '.', '.', '.', '.', '█', '.', '.'],
    ['█', '.', '█', '█', '.', '█', '.', '.', '.', 'E']
])

# Initialize Q-table
Q = {}
for i in range(10):
    for j in range(10):
        Q[(i, j)] = [0, 0, 0, 0]  # 4 possible moves

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor
epsilon = 0.1  # Exploration-exploitation tradeoff

for episode in range(500):
    state = env.reset()
    done = False
    
    while not done:

        if random.uniform(0,1)<epsilon:
            action = random.choice(list(ACTIONS.keys()))
        else:
            action = np.argmax(Q[state])
        

        new_state, reward, done = env.step(action)

        Q[state][action] = Q[state][action] + alpha * (
            reward + gamma * max(Q[new_state]) - Q[state][action]
        )

        state = new_state

    if episode % 100 == 0:
        print(f"Episode {episode} completed.")

print("Training complete!")

for i in range(10):
    for j in range(10):
        print("%.2f" % max(Q[(i,j)]), end=' ')
    print()