# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/12/10 14:04
"""
One-Hot Encoding of Nominal Values
Write a Python function to perform one-hot encoding of nominal values.
The function should take in a 1D numpy array x of integer values and an optional integer n_col representing the number of columns for the one-hot encoded array.
If n_col is not provided, it should be automatically determined from the input array.
Example
Example:
    x = np.array([0, 1, 2, 1, 0])
    output = to_categorical(x)
    print(output)
    # Output:
    # [[1. 0. 0.]
    #  [0. 1. 0.]
    #  [0. 0. 1.]
    #  [0. 1. 0.]
    #  [1. 0. 0.]]

    Reasoning:
    Each element in the input array is transformed into a one-hot encoded vector,
    where the index corresponding to the value in the input array is set to 1,
    and all other indices are set to 0.
"""

import numpy as np


def to_categorical(x, n_col=None):
    # Your code here
    cates = n_col if n_col else len(set(x))
    one_hot_X = []
    for i in range(len(x)):
        i_x = [0] * cates
        i_x[x[i]] = 1
        one_hot_X.append(i_x)

    return one_hot_X

x = np.array([0, 1, 2, 1, 0])
output = to_categorical(x)
print(output)

"""
Test Case 1: Accepted
Input:
print(to_categorical(np.array([0, 1, 2, 1, 0])))
Output:
[[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0]]
Expected:
[[1., 0., 0.], [0., 1., 0.], [0., 0., 1.], [0., 1., 0.], [1., 0., 0.]]
Test Case 2: Accepted
Input:
print(to_categorical(np.array([3, 1, 2, 1, 3]), 4))
Output:
[[0, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]
Expected:
[[0., 0., 0., 1.], [0., 1., 0., 0.], [0., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]]

import numpy as np

def to_categorical(x, n_col=None):
    # One-hot encoding of nominal values
    # If n_col is not provided, determine the number of columns from the input array
    if not n_col:
        n_col = np.amax(x) + 1
    # Initialize a matrix of zeros with shape (number of samples, n_col)
    one_hot = np.zeros((x.shape[0], n_col))
    # Set the appropriate elements to 1
    one_hot[np.arange(x.shape[0]), x] = 1
    return one_hot
    
"""
