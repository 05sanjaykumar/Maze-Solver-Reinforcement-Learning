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

grid = [
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
]


env = MazeEnv(grid)
print("Starting Position:", env.reset())
new_state, reward, done = env.step(3)
print("New State:", new_state, "Reward:", reward, "Done:", done)