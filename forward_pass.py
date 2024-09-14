import numpy as np


# ReLU activation function
def relu(z):
    return np.maximum(0, z)


# Forward pass for the first network (Single-Layer)
def forward_pass_single_layer(x, W):
    z = np.dot(W, x)  # z = Wx
    a = relu(z)  # Apply ReLU activation
    y_hat = a  # y^ is the same as activation output for single-layer
    return a, y_hat


# Forward pass for the second network (Two-Hidden Layers)
def forward_pass_two_layer(x, W1, W2):
    z1 = np.dot(W1, x)  # First layer: z1 = W1x
    a1 = relu(z1)  # Apply ReLU activation for first layer
    z2 = np.dot(W2, a1)  # Second layer: z2 = W2a1
    a2 = relu(z2)  # Apply ReLU activation for second layer
    y_hat = a2  # y^ is the final output
    return a1, a2, y_hat


# Randomly initialize input, weights
np.random.seed(42)  # Set seed for reproducibility
x = np.random.randn(4)  # Input vector (for 4 input neurons)
W_single = np.random.randn(1, 4)  # Weights for single-layer network (1 output neuron)
W1 = np.random.randn(3, 4)  # Weights for first layer of 2-layer network (3 neurons in hidden layer 1)
W2 = np.random.randn(1, 3)  # Weights for second layer of 2-layer network (1 output neuron)

# Forward pass for the single-layer network
a_single, y_hat_single = forward_pass_single_layer(x, W_single)

# Forward pass for the two-hidden-layer network
a1_two_layer, a2_two_layer, y_hat_two_layer = forward_pass_two_layer(x, W1, W2)

# Print the results
print("Single-Layer Network")
print("Activation:", a_single)
print("Prediction (y^):", y_hat_single)
print("\nTwo-Hidden-Layer Network")
print("Activation at Hidden Layer 1:", a1_two_layer)
print("Activation at Hidden Layer 2:", a2_two_layer)
print("Prediction (y^):", y_hat_two_layer)
