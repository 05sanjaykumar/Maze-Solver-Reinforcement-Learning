import numpy as np
import random

class MazeEnv:
    def __init__(self, grid=None):
        self.size = 10
        self.start = (0, 0)
        self.end = (9, 9)
        
        if grid is None:
            self.grid = self.generate_maze()
        else:
            self.grid = grid
        
        self.agent_pos = self.start

    def generate_maze(self):
        # Generate a random maze like before
        grid = [['.' for _ in range(self.size)] for _ in range(self.size)]
        for _ in range(30):  # number of obstacles
            x, y = random.randint(0, 9), random.randint(0, 9)
            if (x, y) not in [self.start, self.end]:
                grid[x][y] = '█'
        grid[self.start[0]][self.start[1]] = 'S'
        grid[self.end[0]][self.end[1]] = 'E'
        return grid

    def reset(self):
        self.agent_pos = self.start
        return self.get_state()

    def get_state(self):
        return np.array(self.agent_pos)

    def step(self, action):
        x, y = self.agent_pos
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = moves[action]
        new_x, new_y = x + dx, y + dy

        # Check for boundaries and obstacles
        if 0 <= new_x < self.size and 0 <= new_y < self.size and self.grid[new_x][new_y] != '█':
            self.agent_pos = (new_x, new_y)
        else:
            return self.get_state(), -1, False  # Hit wall

        if self.agent_pos == self.end:
            return self.get_state(), 10, True  # Reached goal

        return self.get_state(), 0, False  # Valid move

    def render(self):
        for i in range(self.size):
            row = ''
            for j in range(self.size):
                if (i, j) == self.agent_pos:
                    row += 'A '
                else:
                    row += self.grid[i][j] + ' '
            print(row)
        print()
