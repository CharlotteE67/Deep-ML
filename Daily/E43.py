# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/12/2 13:24
"""
Implement Ridge Regression Loss Function
Write a Python function `ridge_loss` that implements the Ridge Regression loss function. The function should take a 2D numpy
array `X` representing the feature matrix, a 1D numpy array `w` representing the coefficients,
a 1D numpy array `y_true` representing the true labels, and a float `alpha` representing the regularization parameter.
The function should return the Ridge loss, which combines the Mean Squared Error (MSE) and a regularization term.
Example
Example:
import numpy as np

X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
w = np.array([0.2, 2])
y_true = np.array([2, 3, 4, 5])
alpha = 0.1

loss = ridge_loss(X, w, y_true, alpha)
print(loss)
# Expected Output: 2.204
"""
import numpy as np


def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    # Your code here
    return sum([(np.dot(w, X[i]) - y_true[i]) ** 2 for i in range(len(y_true))])/len(y_true) + alpha * sum([w_i ** 2 for w_i in w])


X = np.array([[1, 1], [2, 1], [3, 1], [4, 1]])
w = np.array([0.2, 2])
y_true = np.array([2, 3, 4, 5])
alpha = 0.1

loss = ridge_loss(X, w, y_true, alpha)
print(loss)

"""
Test Case 1: Accepted
Input:
X = np.array([[1,1],[2,1],[3,1],[4,1]])
W = np.array([.2,2])
y = np.array([2,3,4,5])
alpha = 0.1
output = ridge_loss(X, W, y, alpha)
print(output)
Output:
2.204
Expected:
2.204
Test Case 2: Accepted
Input:
X = np.array([[1,1,4],[2,1,2],[3,1,.1],[4,1,1.2],[1,2,3]])
W = np.array([.2,2,5])
y = np.array([2,3,4,5,2])
alpha = 0.1
output = ridge_loss(X, W, y, alpha)
print(output)
Output:
164.402
Expected:
164.402

import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:
    loss = np.mean((y_true - X @ w)**2) + alpha * np.sum(w**2)
    return loss

"""