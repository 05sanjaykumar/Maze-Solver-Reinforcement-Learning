import numpy as np
import random

class MazeEnv:
    def __init__(self,grid):
        self.grid = grid
        self.start = (0, 0)
        self.end = (9, 9)
        self.reset()

    def reset(self):
        self.agent_pos = self.start
        return self.get_state()
    
    def get_state(self):

        state = np.zeros((10,10),dtype=np.float32)
        for i in range(10):
            for j in range(10):
                if self.grid[i][j] == '█':
                    state[i][j] = -1  # Wall
        x, y = self.agent_pos
        state[x][y] = 1
        return state.flatten()
    
    def step(self,action):
        x,y = self.agent_pos
        moves = [(x+1,y), (x,y+1), (x-1,y), (x,y-1)]
        nx,ny = moves[action]

        if 0 <= nx < 10 and 0 <= ny < 10 and self.grid[nx][ny] != '█':
            self.agent_pos = (nx, ny)
        
        done = self.agent_pos == self.end
        reward = 1 if done else -0.01  # Encourage faster solutions
        return self.get_state(), reward, done