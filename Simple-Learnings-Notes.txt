import numpy as np

# Initialize the Q-table with all zeros
Q = np.zeros((10, 10, 4))  # 10x10 grid, 4 possible actions (up, down, left, right)

print(Q.shape) 