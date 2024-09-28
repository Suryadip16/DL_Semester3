import numpy as np


# loss function
def loss_fn(y_hat, y):
    return 0.5 * ((y_hat - y) ** 2)


# ReLU function and its derivative
def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return np.where(x > 0, 1, 0)


def forward_pass(x1, x2, w, b1, b2, b3):
    h1z = x1 * w[0, 0] + x2 * w[2, 0] + b1
    h1a = relu(h1z)

    h2z = x1 * w[1, 0] + x2 * w[3, 0] + b2
    h2a = relu(h2z)

    yz = h1a * w[4, 0] + h2a * w[5, 0] + b3
    y = relu(yz)

    return y, h1z, h2z, yz


def backprop_out_layer(alpha, old_w, y_hat, y, yz, x):
    grad = (y - y_hat) * relu_derivative(yz) * x
    new_w = old_w - alpha * (grad)
    return new_w


def backprop_lastbutone_layer(alpha, old_w, y_hat, y, yz, hz, x, w_associated):
    grad = (y - y_hat) * relu_derivative(yz) * w_associated * relu_derivative(hz) * x
    new_w = old_w - alpha * (grad)
    return new_w


# xor truth table
x1 = [0, 0, 1, 1]
x2 = [0, 1, 0, 1]
y = [0, 1, 1, 0]

# assign biases
b1 = 0.01
b2 = 0.01
b3 = 0.01

# assign random weights
w = np.random.rand(6).reshape(6, 1)

epochs = 6000
alpha = 0.01
for i in range(epochs):
    print(f"Epoch: {i + 1}")
    for a, b, c in zip(x1, x2, y):
        y_pred, h1z, h2z, yz = forward_pass(a, b, w, b1, b2, b3)
        loss = loss_fn(c, y_pred)
        print(f"Training Loss: {loss}")

        # backprop for b3:
        new_b3 = backprop_out_layer(alpha, b3, c, y_pred, yz, x=1)

        # backprop for w5
        new_w5 = backprop_out_layer(alpha, w[4, 0], c, y_pred, yz, relu(h1z))

        # backprop for w6
        new_w6 = backprop_out_layer(alpha, w[5, 0], c, y_pred, yz, relu(h2z))

        # backprop for w1
        new_w1 = backprop_lastbutone_layer(alpha, w[0, 0], c, y_pred, yz, h1z, a, w[4, 0])

        # backprop for w2
        new_w2 = backprop_lastbutone_layer(alpha, w[1, 0], c, y_pred, yz, h2z, a, w[5, 0])

        # backprop for w3
        new_w3 = backprop_lastbutone_layer(alpha, w[2, 0], c, y_pred, yz, h1z, b, w[4, 0])

        # backprop for w4
        new_w4 = backprop_lastbutone_layer(alpha, w[3, 0], c, y_pred, yz, h2z, b, w[5, 0])

        # backprop for b1
        new_b1 = backprop_lastbutone_layer(alpha, b1, c, y_pred, yz, h1z, 1, w[4, 0])

        # backprop for b2
        new_b2 = backprop_lastbutone_layer(alpha, b2, c, y_pred, yz, h2z, 1, w[5, 0])

        b1, b2, b3 = new_b1, new_b2, new_b3

        new_w = np.array([new_w1, new_w2, new_w3, new_w4, new_w5, new_w6]).reshape(6, 1)

        w = new_w
    print(f"End of Epoch {i + 1}")
    print("-----------------------------------------------------------------------------------")

print(f"Final Loss: {loss}")






















