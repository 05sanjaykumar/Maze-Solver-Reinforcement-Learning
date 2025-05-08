# Maze Solver using Reinforcement Learning

This project implements a **Reinforcement Learning-based maze solver** using various machine learning algorithms. It aims to create an AI agent that can navigate through a grid (maze) environment by learning from its interactions with the environment. The project follows a structured approach, starting from **Q-learning**, progressing through **Deep Q-learning**, and plans to explore **more complex environments** and techniques in the future.

---

## Table of Contents

* [Introduction](#introduction)
* [Technologies Used](#technologies-used)
* [Project Structure](#project-structure)
* [Key Steps Taken](#key-steps-taken)

  * [Q-Learning](#q-learning)
  * [Deep Q-Learning](#deep-q-learning)
* [What We Have Done So Far](#what-we-have-done-so-far)
* [What’s Next](#whats-next)
* [Getting Started](#getting-started)
* [License](#license)

---

## Introduction

In this project, we explore the problem of teaching an agent to navigate a maze using **Reinforcement Learning**. The agent will learn the optimal policy to move from the starting point to the goal in the least amount of time. Initially, we start with the basic **Q-learning** algorithm, and as we move forward, we build more complex models, including **Deep Q-learning**, to solve the problem more efficiently.

---

## Technologies Used

* **Python**: The main programming language.
* **PyTorch**: For creating and training the neural network in **Deep Q-learning**.
* **NumPy**: For handling arrays and matrix operations.
* **OpenAI Gym (future)**: For building and testing reinforcement learning environments.

---

## Project Structure

* `Q_learning.py`: Implements the basic **Q-learning** algorithm.
* `Deep_Q_learning.py`: Implements the **Deep Q-learning** algorithm using PyTorch.
* `Maze.py`: Contains the maze environment that the agent interacts with.
* `README.md`: Project documentation.

---

## Key Steps Taken

### Q-Learning

* **Q-Learning** is a simple, tabular method of solving reinforcement learning problems. It builds a Q-table to store **state-action values**, which represents the expected cumulative reward for taking an action in a given state.
* **Core idea**: The agent updates the Q-table based on its actions and rewards, gradually learning which actions are optimal.
* Implemented a basic **grid maze** where the agent learns to navigate from the start position to the goal using Q-learning.

### Deep Q-Learning

* **Deep Q-Learning** (DQN) combines Q-learning with neural networks to handle large and complex state spaces, where Q-tables would be inefficient.
* Implemented a **neural network** to approximate the Q-function, and used **experience replay** and **target networks** for stability during training.
* Replaced the Q-table with a neural network that learns to predict the Q-values for each action.

---

## What We Have Done So Far

1. **Implemented Q-learning**:

   * Built a **Q-table** for state-action values.
   * Trained the agent to solve a simple maze using Q-learning.
2. **Built a neural network for Deep Q-learning**:

   * Created a **Q-network** using PyTorch.
   * Started integrating **Deep Q-learning** to replace the Q-table and improve performance.
3. **Maze Environment**:

   * Developed a simple **grid-based maze environment** where the agent can take actions and receive rewards.
   * The agent is trained to find the shortest path to the goal.

---

## What’s Next

1. **Improvement in Maze Complexity**:

   * Expanding the environment to include **larger mazes** with more obstacles.
   * Adding **dynamic rewards** and more complex state-action spaces.

2. **Exploration of Different RL Algorithms**:

   * We will dive deeper into **Deep Q-learning** and explore methods like **Double DQN** and **Prioritized Experience Replay**.
   * Investigating **Policy Gradient Methods** like **REINFORCE** and **Actor-Critic** algorithms to further enhance the agent’s performance.

3. **Integration with OpenAI Gym**:

   * Moving the project to **OpenAI Gym** for easier testing and scalability.
   * Adding new environments and challenges to the agent’s learning pipeline.

4. **Real-World Applications**:

   * Applying reinforcement learning to more **real-world problems** like robotics and autonomous vehicles, using the maze-solving framework as the initial step.

---

## Getting Started

### Prerequisites

To run the project, you’ll need the following:

* **Python 3.x** installed.
* Required Python libraries:

  * `numpy`
  * `torch` (PyTorch)

You can install the necessary packages using `pip`:

```bash
pip install -r requirements.txt
```

### Running the Project

1. Clone the repository:

```bash
git clone https://github.com/05sanjaykumar/Maze-Solver-Reinforcement-Learning.git
```

2. Navigate to the project directory:

```bash
cd Maze-Solver-Reinforcement-Learning
```

3. Run the Q-learning script or the Deep Q-learning script:

```bash
python Q_learning.py
# or
python Deep_Q_learning.py
```

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

> [!NOTE]  
> Under Development 
