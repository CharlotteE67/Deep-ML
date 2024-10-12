"""
Single Neuron (easy)
Write a Python function that simulates a single neuron with a sigmoid activation function for binary classification,
handling multidimensional input features. The function should take a list of feature vectors
(each vector representing multiple features for an example), associated true binary labels,
and the neuron's weights (one for each feature) and bias as input. It should return the predicted probabilities
after sigmoid activation and the mean squared error between the predicted probabilities and the true labels,
both rounded to four decimal places.
Example
Example:
input: features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], labels = [0, 1, 0], weights = [0.7, -0.4], bias = -0.1
output: ([0.4626, 0.4134, 0.6682], 0.3349)
reasoning: For each input vector, the weighted sum is calculated by multiply
"""


import math

from typing import List


def single_neuron_model(features: List[List[float]], labels: List[int], weights: List[float], bias: float) -> (
List[float], float):
    # Your code here
    z = []
    for i in range(len(features)):
        z_i = 0
        for j in range(len(features[i])):
            z_i += features[i][j] * weights[j]
        z_i += + bias
        z.append(z_i)
    probabilities = []
    for zi in z:
        probabilities.append(1/(1+math.exp(-zi)))
    mse = sum([(probabilities[i] - labels[i])**2 for i in range(len(labels))]) / len(probabilities)
    return probabilities, mse

features = [[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]]
labels = [0, 1, 0]
weights = [0.7, -0.4]
bias = -0.1

print(single_neuron_model(features, labels, weights, bias))


"""
Test Case 1: Accepted
Input:
print(single_neuron_model([[0.5, 1.0], [-1.5, -2.0], [2.0, 1.5]], [0, 1, 0], [0.7, -0.4], -0.1))
Output:
([0.46257015465625034, 0.41338242108267, 0.668187772168166], 0.33485541024953136)
Expected:
([0.4626, 0.4134, 0.6682], 0.3349)
Test Case 2: Accepted
Input:
print(single_neuron_model([[1, 2], [2, 3], [3, 1]], [1, 0, 1], [0.5, -0.2], 0))
Output:
([0.5249791874789399, 0.598687660112452, 0.7858349830425586], 0.2099794470624907)
Expected:
([0.525, 0.5987, 0.7858], 0.21)

import math
def single_neuron_model(features, labels, weights, bias):
    probabilities = []
    for feature_vector in features:
        z = sum(weight * feature for weight, feature in zip(weights, feature_vector)) + bias
        prob = 1 / (1 + math.exp(-z))
        probabilities.append(round(prob, 4))
    
    mse = sum((prob - label) ** 2 for prob, label in zip(probabilities, labels)) / len(labels)
    mse = round(mse, 4)
    
    return probabilities, mse
"""