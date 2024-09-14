import numpy as np
import matplotlib.pyplot as plt

# Generate 100 equally spaced values between -10 and 10
z = np.linspace(-10, 10, 100)


# Sigmoid function and its derivative
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_derivative(x):
    s = sigmoid(x)
    return s * (1 - s)


# Tanh function and its derivative
def tanh(x):
    return np.tanh(x)


def tanh_derivative(x):
    return 1 - np.tanh(x) ** 2


# ReLU function and its derivative
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


# Leaky ReLU function and its derivative
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)


def leaky_relu_derivative(x, alpha=0.01):
    return np.where(x > 0, 1, alpha)


# Softmax function and its derivative
def softmax(x):
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x)


# Plotting the functions and their derivatives
functions = {
    'Sigmoid': (sigmoid, sigmoid_derivative),
    'Tanh': (tanh, tanh_derivative),
    'ReLU': (relu, relu_derivative),
    'Leaky ReLU': (leaky_relu, leaky_relu_derivative),
}

fig, axs = plt.subplots(len(functions), 2, figsize=(10, 15))

for i, (name, (func, deriv)) in enumerate(functions.items()):
    axs[i, 0].plot(z, func(z))
    axs[i, 0].set_title(f'{name} Function')
    axs[i, 1].plot(z, deriv(z))
    axs[i, 1].set_title(f'{name} Derivative')
plt.show()
# Softmax has a different scale, hence plotted separately

plt.plot(z, softmax(z))
plt.title('Softmax Function')

plt.tight_layout()
plt.show()
