# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/11 16:24
"""

Single Neuron with Backpropagation (medium) Write a Python function that simulates a single neuron with sigmoid
activation, and implements backpropagation to update the neuron's weights and bias. The function should take a list
of feature vectors, associated true binary labels, initial weights, initial bias, a learning rate, and the number of
epochs. The function should update the weights and bias using gradient descent based on the MSE loss, and return the
updated weights, bias, and a list of MSE values for each epoch, each rounded to four decimal places.
Example Example:
input: features = [[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]], labels = [1, 0, 0], initial_weights = [0.1, -0.2],
initial_bias = 0.0, learning_rate = 0.1, epochs = 2
output: updated_weights = [0.1036, -0.1425], updated_bias =
-0.0167, mse_values =[0.3033, 0.2942]
reasoning: The neuron receives feature vectors and computes predictions using
the sigmoid activation. Based on the predictions and true labels, the gradients of MSE loss with respect to weights
and bias are computed and used to update the model parameters across epochs.
"""

import numpy as np
from typing import List


def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float,
                 learning_rate: float, epochs: int) -> (np.ndarray, float, List[float]):
    # Your code here
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    updated_weights, updated_bias, mse_values = initial_weights, initial_bias, []
    for _ in range(epochs):
        # print(np.dot(features, updated_weights))
        z = np.dot(features, updated_weights) + updated_bias
        out = sigmoid(z)
        # print(out)
        # print(np.dot(features, updated_weights) + updated_bias)
        # print((out - labels) ** 2)
        mse_loss = np.mean((out - labels) ** 2)
        # print(mse_loss)
        # print(np.dot(features.T, ((out - labels) * out * (1 - out))))
        updated_weights = updated_weights - learning_rate * (
                2 / len(labels) * np.dot(features.T, ((out - labels) * out * (1 - out))))
        updated_bias = updated_bias - learning_rate * np.sum(2 / len(labels) * ((out - labels) * out * (1 - out)))
        # print(updated_weights, updated_bias)

        updated_weights, updated_bias, mse_loss = np.round(updated_weights, 4), round(updated_bias, 4), round(mse_loss, 4)
        mse_values.append(mse_loss)

    return updated_weights, updated_bias, mse_values


# import numpy as np
#
#
# def sigmoid(x):
#     return 1 / (1 + np.exp(-x))
#
#
# def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
#     weights = np.array(initial_weights)
#     bias = initial_bias
#     features = np.array(features)
#     labels = np.array(labels)
#     mse_values = []
#
#     for _ in range(epochs):
#         z = np.dot(features, weights) + bias
#         predictions = sigmoid(z)
#
#         mse = np.mean((predictions - labels) ** 2)
#         mse_values.append(round(mse, 4))
#
#         # Gradient calculation for weights and bias
#         errors = predictions - labels
#         weight_gradients = (2 / len(labels)) * np.dot(features.T, errors * predictions * (1 - predictions))
#         bias_gradient = (2 / len(labels)) * np.sum(errors * predictions * (1 - predictions))
#
#         # Update weights and bias
#         weights -= learning_rate * weight_gradients
#         bias -= learning_rate * bias_gradient
#
#         # Round weights and bias for output
#         updated_weights = np.round(weights, 4)
#         updated_bias = round(bias, 4)
#
#     return updated_weights.tolist(), updated_bias, mse_values


features = np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]])
labels = np.array([1, 0, 0])
initial_weights = np.array([0.1, -0.2])
initial_bias = 0.0
learning_rate = 0.1
epochs = 2

print(train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs))

"""
Test Case 1: Accepted
Input:
print(train_neuron(np.array([[1.0, 2.0], [2.0, 1.0], [-1.0, -2.0]]), np.array([1, 0, 0]), np.array([0.1, -0.2]), 0.0, 0.1, 2))
Output:
(array([ 0.1036, -0.1426]), -0.0167, [0.3033, 0.2942])
Expected:
([0.1036, -0.1425], -0.0167, [0.3033, 0.2942])
Test Case 2: Accepted
Input:
print(train_neuron(np.array([[1, 2], [2, 3], [3, 1]]), np.array([1, 0, 1]), np.array([0.5, -0.2]), 0, 0.1, 3))
Output:
(array([ 0.4892, -0.2301]), 0.0029, [0.21, 0.2087, 0.2076])
Expected:
([0.4892, -0.2301], 0.0029, [0.21, 0.2087, 0.2076])

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def train_neuron(features, labels, initial_weights, initial_bias, learning_rate, epochs):
    weights = np.array(initial_weights)
    bias = initial_bias
    features = np.array(features)
    labels = np.array(labels)
    mse_values = []

    for _ in range(epochs):
        z = np.dot(features, weights) + bias
        predictions = sigmoid(z)
        
        mse = np.mean((predictions - labels) ** 2)
        mse_values.append(round(mse, 4))

        # Gradient calculation for weights and bias
        errors = predictions - labels
        weight_gradients = (2/len(labels)) * np.dot(features.T, errors * predictions * (1 - predictions))
        bias_gradient = (2/len(labels)) * np.sum(errors * predictions * (1 - predictions))
        
        # Update weights and bias
        weights -= learning_rate * weight_gradients
        bias -= learning_rate * bias_gradient

        # Round weights and bias for output
        updated_weights = np.round(weights, 4)
        updated_bias = round(bias, 4)

    return updated_weights.tolist(), updated_bias, mse_values
    

"""