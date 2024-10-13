import numpy as np


class NeuralNetworkWithDropout:
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.5):
        # Initialize weights and biases for two layers (input -> hidden, hidden -> output)
        # Weights are randomly initialized, biases are set to zero
        self.W1 = np.random.randn(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        # Set the dropout rate (fraction of neurons to drop)
        self.dropout_rate = dropout_rate

    def relu(self, x):
        # ReLU activation function: returns max(0, x)
        return np.maximum(0, x)

    def relu_derivative(self, x):
        # Derivative of ReLU: returns 1 where x > 0, otherwise 0
        return np.where(x > 0, 1, 0)

    def softmax(self, x):
        # Softmax activation function for multi-class classification
        # Computes exponentials and normalizes them to sum to 1
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # Subtract max for numerical stability
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X, training=True):
        # Forward pass: computes activations for each layer

        # Step 1: Compute z1 (linear combination of input and weights for hidden layer)
        z1 = X.dot(self.W1) + self.b1

        # Step 2: Apply ReLU activation function
        a1 = self.relu(z1)

        # Step 3: Apply dropout during training
        if training:
            # Create a dropout mask: randomly set some activations to 0 based on dropout rate
            # Scale remaining activations by 1/(1-dropout_rate) to maintain output scale
            self.dropout_mask = (np.random.rand(*a1.shape) > self.dropout_rate) / (1 - self.dropout_rate)
            # Apply the mask to the activations
            a1 *= self.dropout_mask

        # Store activations of the hidden layer (useful for backpropagation)
        self.a1 = a1

        # Step 4: Compute z2 (linear combination of hidden layer activations and weights for output layer)
        z2 = a1.dot(self.W2) + self.b2

        # Step 5: Apply softmax activation function for output
        a2 = self.softmax(z2)

        return a2

    def backward(self, X, y, output, learning_rate=0.01):
        # Backpropagation: compute gradients and update weights and biases

        # Number of samples
        m = X.shape[0]

        # Compute gradient of the loss with respect to z2 (output layer pre-activation)
        dz2 = output - y  # Derivative of softmax cross-entropy loss

        # Compute gradients for weights and biases of the output layer
        dW2 = (self.a1.T).dot(dz2) / m  # Gradient for W2
        db2 = np.sum(dz2, axis=0, keepdims=True) / m  # Gradient for b2

        # Backpropagate through the hidden layer
        da1 = dz2.dot(self.W2.T)  # Gradient of the loss with respect to hidden layer activations
        dz1 = da1 * self.relu_derivative(self.a1)  # Apply ReLU derivative

        # Apply dropout mask to the gradient (only backpropagate through non-dropped neurons)
        dz1 *= self.dropout_mask

        # Compute gradients for weights and biases of the hidden layer
        dW1 = (X.T).dot(dz1) / m  # Gradient for W1
        db1 = np.sum(dz1, axis=0, keepdims=True) / m  # Gradient for b1

        # Update weights and biases using gradient descent
        self.W1 -= learning_rate * dW1
        self.b1 -= learning_rate * db1
        self.W2 -= learning_rate * dW2
        self.b2 -= learning_rate * db2

    def compute_loss(self, y, y_hat):
        # Compute the cross-entropy loss
        # y: true labels (one-hot encoded), y_hat: predicted probabilities
        m = y.shape[0]
        loss = -np.sum(y * np.log(y_hat)) / m  # Cross-entropy formula
        return loss


# Example usage
np.random.seed(42)  # For reproducibility

# Generate random input data (100 samples, 20 features) and one-hot encoded labels (10 classes)
X = np.random.rand(100, 20)  # 100 samples, 20 features
y = np.eye(10)[np.random.choice(10, 100)]  # 100 samples, 10 classes (one-hot encoded)

# Initialize neural network with dropout (input size: 20, hidden layer: 50 neurons, output: 10 classes)
nn = NeuralNetworkWithDropout(input_size=20, hidden_size=50, output_size=10, dropout_rate=0.5)

# Perform a forward pass (training mode: dropout applied)
output = nn.forward(X, training=True)

# Compute the initial loss
loss = nn.compute_loss(y, output)
print(f"Initial loss: {loss}")
