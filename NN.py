# the raw version of this code was implemented by me from scratch, it had some errors that were resolved using perplexing.ai
import numpy as np
import pandas as pd
from scipy.special import softmax

# Load the MNIST dataset
class NN:
    def __init__(self, X, y, train, alpha = 0.1, iter = 1000) :
        self.X = X
        self.y = y
        self.iter = iter
        X = (X - np.mean(X)) / np.std(X)
        self.alpha = alpha
        params = []
        W1 = np.random.randn(15, 30) * np.sqrt(2. / 30)  # Use np.random.randn for normal distribution
        B1 = np.zeros((1, 15)) * np.sqrt(2. / 1)  # Initialize biases to zero
        W2 = np.random.randn(2, 15) * np.sqrt(2. / 15)  # Adjusted for the number of neurons in the previous layer
        B2 = np.zeros((1, 2)) * np.sqrt(2. / 1) # Initialize biases to zero
        params.extend([W1, B1, W2, B2])
        if train:
            params = self.gradient_descent(iter, W1, W2, B1, B2, alpha, self.X, self.y)
            return params
        else :
            _, _, _, A2 = self.forward_prop(W1, W2, B1, B2, self.X)
            predictions = self.get_predictions(A2)
            return predictions

        
    def ReLU(self, Z):
        return np.maximum(Z, 0)

    def deriv_relu(self, Z):
        return Z > 0

    def compute_softmax(self, Z):
        return softmax(Z, axis=1)

    def forward_prop(self, W1, W2, B1, B2, X):
        Z1 = np.dot(X, W1.T) + B1
        A1 = self.ReLU(Z1)
        Z2 = np.dot(A1, W2.T) + B2
        A2 = self.compute_softmax(Z2)
        return Z1, Z2, A1, A2

    def one_hot(self, y):
        arr = np.zeros((len(y), 2))
        for i in range(len(y)):
            index = int(y[i])  # No need to subtract 1
            arr[i, index] = 1
        return arr

    def backward_prop(self, alpha, A2, A1, Z1, Z2, W2, W1, B2, B1, y):
        m = A1.shape[0]
        dA2 = A2 - self.one_hot(y)
        dZ2 = dA2  # Softmax derivative
        dW2 = dZ2.T.dot(A1) / m  # Average over batch size
        dB2 = np.mean(dZ2, axis=0, keepdims=True)
        dA1 = dZ2.dot(W2)
        dZ1 = dA1 * self.deriv_relu(Z1)
        dW1 = dZ1.T.dot(self.X) / m  # Average over batch size
        dB1 = np.mean(dZ1, axis=0, keepdims=True)

        W1 -= alpha * dW1
        B1 -= alpha * dB1
        W2 -= alpha * dW2
        B2 -= alpha * dB2

        return W1, B1, W2, B2

    def get_accuracy(self, predictions, Y):
        return np.sum(predictions == Y) / len(Y)

    def get_predictions(self, Z):
        max_indices = np.argmax(Z, axis=1)
        return max_indices

    def gradient_descent(self, iter, W1, W2, B1, B2, alpha, X, y):
        for i in range(iter):
            Z1, Z2, A1, A2 = self.forward_prop(W1, W2, B1, B2, X)
            W1, B1, W2, B2 = self.backward_prop(alpha, A2, A1, Z1, Z2, W2, W1, B2, B1, y)
            
            predictions = self.get_predictions(A2)
            accuracy = self.get_accuracy(predictions, y)
            loss = -np.mean(np.sum(self.one_hot(y) * np.log(A2 + 1e-12), axis=1))  # Cross-entropy loss
            
            if i % 100 == 0:  # Print every 100 iterations
                print(f"Iteration {i}, Accuracy: {accuracy:.4f}, Loss: {loss:.4f}")

        return W1, B1, W2, B2

