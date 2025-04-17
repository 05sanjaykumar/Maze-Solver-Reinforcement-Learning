#just how neural network works actually


# Layer 1: Input → Hidden1 (2 neurons)
W1 = np.random.randn(2, 2)  # (input_size, hidden1_size)
b1 = np.zeros((1, 2))

# Layer 2: Hidden1 → Hidden2 (e.g., 3 neurons)
W2 = np.random.randn(2, 3)  # (hidden1_size, hidden2_size)
b2 = np.zeros((1, 3))

# Layer 3: Hidden2 → Output (1 neuron)
W3 = np.random.randn(3, 1)  # (hidden2_size, output_size)
b3 = np.zeros((1, 1))


#Notes, 2 input, two input neuron, and if one output then output neuron
# If we have one input, then if we need three output, we need to have one neuron, Just see below

W1 = np.random.randn(2, 1) 
b1 = np.zeros((1, 1))


W2 = np.random.randn(2, 3)  # three outputs
b2 = np.zeros((1, 3)) 