# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/14 15:03
"""
Implementation of Log Softmax Function
In machine learning and statistics, the softmax function is a generalization of the logistic function that converts a vector of scores into probabilities.
The log-softmax function is the logarithm of the softmax function, and it is often used for numerical stability when computing the softmax of large numbers.
Given a 1D numpy array of scores, implement a Python function to compute the log-softmax of the array.
Example
Example:
A = np.array([1, 2, 3])
print(log_softmax(A))

Output:
array([-2.4076, -1.4076, -0.4076])
"""
import numpy as np


def log_softmax(scores: list) -> np.ndarray:
    # Your code here
    scores = np.array(scores, dtype=float)
    scores = scores - max(scores)
    log = np.log(sum([np.exp(s) for s in scores]))
    scores -= log
    return scores


A = np.array([1, 2, 3])
print(log_softmax(A))

"""
Test Case 1: Accepted
Input:
print(log_softmax([1, 2, 3]))
Output:
[-2.40760596 -1.40760596 -0.40760596]
Expected:
[-2.4076, -1.4076, -0.4076]
Test Case 2: Accepted
Input:
print(log_softmax([1, 1, 1]))
Output:
[-1.09861229 -1.09861229 -1.09861229]
Expected:
[-1.0986, -1.0986, -1.0986]
Test Case 3: Accepted
Input:
print(log_softmax([1, 1, .0000001]))
Output:
[-0.86199482 -0.86199482 -1.86199472]
Expected:
[-0.862, -0.862, -1.862]


log softmax(x_i) = x_i - max(x) - log(\sum_{j=1}^n e^{x_j - max(x)})

import numpy as np

def log_softmax(scores: list) -> np.ndarray:
    # Subtract the maximum value for numerical stability
    scores = scores - np.max(scores)
    return scores - np.log(np.sum(np.exp(scores)))
"""
