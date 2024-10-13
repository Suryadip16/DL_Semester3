import numpy as np


class BatchNormalization:
    def __init__(self, num_features, epsilon=1e-5, momentum=0.9):
        # Initialize parameters for batch normalization
        self.gamma = np.ones((1, num_features))  # Scale parameter
        self.beta = np.zeros((1, num_features))  # Shift parameter
        self.epsilon = epsilon  # Small constant to avoid division by zero
        self.momentum = momentum  # Momentum for running mean/variance during inference
        # Initialize running mean and variance for inference (used during testing)
        self.running_mean = np.zeros((1, num_features))
        self.running_var = np.ones((1, num_features))

    def forward(self, X, training=True):
        if training:
            # Compute mean and variance across the batch (axis=0)
            batch_mean = np.mean(X, axis=0, keepdims=True)
            batch_var = np.var(X, axis=0, keepdims=True)

            # Normalize the batch
            self.X_centered = X - batch_mean
            self.std = np.sqrt(batch_var + self.epsilon)
            X_norm = self.X_centered / self.std

            # Scale and shift the normalized values
            out = self.gamma * X_norm + self.beta

            # Update running mean and variance for inference
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var
        else:
            # Use running mean and variance during inference
            X_norm = (X - self.running_mean) / np.sqrt(self.running_var + self.epsilon)
            out = self.gamma * X_norm + self.beta

        return out

    def backward(self, d_out):
        # Backpropagation through batch normalization
        N, D = d_out.shape

        # Gradients with respect to gamma and beta
        dgamma = np.sum(d_out * (self.X_centered / self.std), axis=0)
        dbeta = np.sum(d_out, axis=0)

        # Backpropagate the normalized inputs
        dX_norm = d_out * self.gamma
        dvar = np.sum(dX_norm * self.X_centered * -0.5 * (self.std ** -3), axis=0)
        dmean = np.sum(dX_norm * -1 / self.std, axis=0) + dvar * np.mean(-2 * self.X_centered, axis=0)

        # Final gradient with respect to input X
        dX = dX_norm / self.std + dvar * 2 * self.X_centered / N + dmean / N

        return dX, dgamma, dbeta


class LayerNormalization:
    def __init__(self, num_features, epsilon=1e-5):
        # Initialize parameters for layer normalization
        self.gamma = np.ones((1, num_features))  # Scale parameter
        self.beta = np.zeros((1, num_features))  # Shift parameter
        self.epsilon = epsilon  # Small constant to avoid division by zero

    def forward(self, X):
        # Compute mean and variance for each training example (axis=1)
        mean = np.mean(X, axis=1, keepdims=True)
        var = np.var(X, axis=1, keepdims=True)

        # Normalize the layer's activations
        X_centered = X - mean
        std = np.sqrt(var + self.epsilon)
        X_norm = X_centered / std

        # Scale and shift the normalized values
        out = self.gamma * X_norm + self.beta

        return out

    def backward(self, d_out, X):
        # Backpropagation through layer normalization
        N, D = X.shape

        # Compute mean and variance for the current layer
        mean = np.mean(X, axis=1, keepdims=True)
        var = np.var(X, axis=1, keepdims=True)
        X_centered = X - mean
        std_inv = 1.0 / np.sqrt(var + self.epsilon)

        # Gradients with respect to gamma and beta
        dgamma = np.sum(d_out * X_centered * std_inv, axis=0)
        dbeta = np.sum(d_out, axis=0)

        # Backpropagate the normalized inputs
        dX_norm = d_out * self.gamma
        dvar = np.sum(dX_norm * X_centered * -0.5 * std_inv ** 3, axis=1, keepdims=True)
        dmean = np.sum(dX_norm * -std_inv, axis=1, keepdims=True) + dvar * np.mean(-2 * X_centered, axis=1,
                                                                                   keepdims=True)

        # Final gradient with respect to input X
        dX = dX_norm * std_inv + dvar * 2 * X_centered / D + dmean / D

        return dX, dgamma, dbeta


# Example input data (batch of size 4, 5 features each)
X_batch = np.random.rand(4, 5)

# Batch Normalization Example
bn = BatchNormalization(num_features=5)
out_bn = bn.forward(X_batch, training=True)  # Forward pass (training mode)
dX_bn, dgamma_bn, dbeta_bn = bn.backward(np.random.rand(4, 5))  # Backward pass

print("Batch Normalization output:", out_bn)

# Layer Normalization Example
ln = LayerNormalization(num_features=5)
out_ln = ln.forward(X_batch)  # Forward pass
dX_ln, dgamma_ln, dbeta_ln = ln.backward(np.random.rand(4, 5), X_batch)  # Backward pass

print("Layer Normalization output:", out_ln)
