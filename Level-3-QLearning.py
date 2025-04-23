#This use Q learning approach and stores in table actually.

import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Step 1: Generate a solvable maze
def create_grid(extra_paths=10):
    grid = [['█' for _ in range(10)] for _ in range(10)]
    path = [(0, 0)]
    visited = set(path)

    while path[-1] != (9, 9):  
        x, y = path[-1]
        neighbors = [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]
        random.shuffle(neighbors)

        for nx, ny in neighbors:
            if 0 <= nx < 10 and 0 <= ny < 10 and (nx, ny) not in visited:
                path.append((nx, ny))
                visited.add((nx, ny))
                break
        else:
            path.pop()

    for x, y in visited:
        grid[x][y] = '.'

    obstacles = [(x, y) for x in range(10) for y in range(10) if grid[x][y] == '█']
    random.shuffle(obstacles)

    for _ in range(extra_paths):
        if obstacles:
            x, y = obstacles.pop()
            grid[x][y] = '.'

    grid[0][0] = 'S'
    grid[9][9] = 'E'
    return grid

# Step 2: Convert grid to a graph
def convert_to_graph(grid):
    graph = {}
    directions = [(1,0), (0,1), (-1,0), (0,-1)]
    for i in range(10):
        for j in range(10):
            if grid[i][j] == '█':
                continue
            arr = []
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0<=ni<10 and 0<=nj<10 and grid[ni][nj]!='█':
                    arr.append((ni,nj))
            graph[(i,j)] = arr
    return graph

# Step 3: Train AI using Q-Learning
def train_q_learning(grid, episodes=5000, alpha=0.1, gamma=0.9, epsilon=0.1):
    states = [(i, j) for i in range(10) for j in range(10) if grid[i][j] != '█']
    actions = [(1,0), (0,1), (-1,0), (0,-1)]
    Q = {state: {action: 0 for action in actions} for state in states}

    for _ in range(episodes):
        state = (0, 0)
        while state != (9, 9):
            if random.uniform(0, 1) < epsilon:
                action = random.choice(actions)  
            else:
                action = max(Q[state], key=Q[state].get)  

            new_state = (state[0] + action[0], state[1] + action[1])
            if new_state not in states:
                continue  

            reward = 100 if new_state == (9, 9) else -1  

            Q[state][action] += alpha * (reward + gamma * max(Q[new_state].values()) - Q[state][action])
            state = new_state

    return Q

# Step 4: AI Solves the Maze
def solve_maze_with_q(Q, start=(0,0), end=(9,9)):
    state = start
    path = [state]
    while state != end:
        action = max(Q[state], key=Q[state].get)
        state = (state[0] + action[0], state[1] + action[1])
        path.append(state)
    return path

# Step 5: Visualizing the AI Path
def animate_solution(grid, path):
    fig, ax = plt.subplots()
    ax.set_xticks(np.arange(-0.5, 10, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 10, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=1)

    display_grid = np.zeros((10, 10))
    for i in range(10):
        for j in range(10):
            if grid[i][j] == "█":
                display_grid[i, j] = 1  

    im = ax.imshow(display_grid, cmap="gray_r")

    def update(frame):
        if frame > 0:
            x, y = path[frame - 1]
            display_grid[x, y] = 0.5  
        x, y = path[frame]
        display_grid[x, y] = 0.2  
        im.set_array(display_grid)
        return [im]

    ani = animation.FuncAnimation(fig, update, frames=len(path), interval=300, repeat=False)
    plt.show()

# Run the AI
grid = create_grid()
graph = convert_to_graph(grid)
Q_table = train_q_learning(grid)
solved_path = solve_maze_with_q(Q_table)
animate_solution(grid, solved_path)
