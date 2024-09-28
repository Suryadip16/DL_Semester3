import numpy as np


def loss_fn(y_hat, y):
    return 0.5 * ((y_hat - y) ** 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def forward_pass(x1, x2, x3, w):
    z = (w[0, 0] * x1) + (w[1, 0] * x2) + (w[2, 0] * x3)
    a = sigmoid(z)
    return a


def backprop(alpha, old_w, y, y_hat, x):
    grad = (y - y_hat) * (y * (1 - y)) * x
    new_w = old_w - (alpha * grad)
    return new_w


# Assign random weights
w = np.random.rand(3).reshape(3, 1)

# Perform forward pass
x1 = [0, 1, 1, 0]
x2 = [0, 1, 0, 1]
x3 = [1, 1, 1, 1]
y = [0, 1, 1, 0]

num_iterations = 10000
alpha = 0.1
for i in range(num_iterations):
    print(f"Epoch {i + 1}")
    for j, k, l, m in zip(x1, x2, x3, y):
        res = forward_pass(j, k, l, w)
        loss = loss_fn(m, res)
        print(loss)
        # Backprop for w1
        new_w1 = backprop(alpha, w[0, 0], res, m, j)

        # Backprop for w2
        new_w2 = backprop(alpha, w[1, 0], res, m, k)

        # Backprop for w3
        new_w3 = backprop(alpha, w[2, 0], res, m, l)

        new_w = np.array([new_w1, new_w2, new_w3]).reshape(3, 1)

        w = new_w

print(loss)
