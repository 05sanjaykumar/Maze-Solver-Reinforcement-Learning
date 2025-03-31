import pygame
import random
import time
from collections import deque

# Constants
WIDTH, HEIGHT = 500, 500
ROWS, COLS = 10, 10
CELL_SIZE = WIDTH // COLS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
YELLOW = (255, 255, 0)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Maze Solver")
clock = pygame.time.Clock()

# Create Grid Without Blocking From Starting To Ending
def create_grid():
    grid = [['█' for _ in range(COLS)] for _ in range(ROWS)]
    path = [(0, 0)]
    visited = set(path)

    while path[-1] != (ROWS - 1, COLS - 1):
        x, y = path[-1]
        neighbors = [(x+1, y), (x, y+1), (x-1, y), (x, y-1)]
        random.shuffle(neighbors)

        for nx, ny in neighbors:
            if 0 <= nx < ROWS and 0 <= ny < COLS and (nx, ny) not in visited:
                path.append((nx, ny))
                visited.add((nx, ny))
                break
        else:
            path.pop()

    for x, y in visited:
        grid[x][y] = '.'
    
    grid[0][0] = 'S'
    grid[ROWS - 1][COLS - 1] = 'E'
    return grid

# Convert to Graph
def convert_to_graph(grid):
    graph = {}
    directions = [(1,0), (0,1), (-1,0), (0,-1)]
    for i in range(ROWS):
        for j in range(COLS):
            if grid[i][j] == '█':
                continue
            arr = []
            for dx, dy in directions:
                ni, nj = i + dx, j + dy
                if 0 <= ni < ROWS and 0 <= nj < COLS and grid[ni][nj] != '█':
                    arr.append((ni, nj))
            graph[(i, j)] = arr
    return graph

# BFS Shortest Path
def bfs_shortest_path(graph, start, end):
    queue = deque([(start, [start])])
    visited = set()
    
    while queue:
        node, path = queue.popleft()
        if node == end:
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor in graph[node]:
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))

    return None

# Draw Grid
def draw_grid(grid):
    for i in range(ROWS):
        for j in range(COLS):
            color = WHITE if grid[i][j] == '.' else BLACK
            if grid[i][j] == 'S':
                color = GREEN
            elif grid[i][j] == 'E':
                color = RED
            pygame.draw.rect(screen, color, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE))
            pygame.draw.rect(screen, YELLOW, (j * CELL_SIZE, i * CELL_SIZE, CELL_SIZE, CELL_SIZE), 1)

# Animate Path
def animate_path(path):
    for x, y in path:
        pygame.draw.rect(screen, BLUE, (y * CELL_SIZE, x * CELL_SIZE, CELL_SIZE, CELL_SIZE))
        pygame.display.update()
        time.sleep(0.1)  

# Main
grid = create_grid()
graph = convert_to_graph(grid)
start, end = (0, 0), (ROWS - 1, COLS - 1)
shortest_path = bfs_shortest_path(graph, start, end)

running = True
while running:
    screen.fill(WHITE)
    draw_grid(grid)

    if shortest_path:
        animate_path(shortest_path)
    
    pygame.display.update()

    # Keep window open
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

pygame.quit()
