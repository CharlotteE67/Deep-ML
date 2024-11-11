# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/11/11 12:57
"""
Batch Iterator for Dataset
Write a Python function to create a batch iterator for the samples in a numpy array X and an optional numpy array y.
The function should yield batches of a specified size. If y is provided, the function should yield batches of (X, y) pairs;
otherwise, it should yield batches of X only.
Example
Example:
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    batch_size = 2
    batch_iterator(X, y, batch_size)
    output:
    [[[[1, 2], [3, 4]], [1, 2]],
     [[[5, 6], [7, 8]], [3, 4]],
     [[[9, 10]], [5]]]

     Reasoning:
    The dataset X contains 5 samples, and we are using a batch size of 2. Therefore, the function will divide the dataset into 3 batches.
    The first two batches will contain 2 samples each, and the last batch will contain the remaining sample.
    The corresponding values from y are also included in each batch.
"""
import numpy as np


def batch_iterator(X, y=None, batch_size=64):
    # Your code here
    batched_data = []
    batch_num = len(X) // batch_size
    for i in range(batch_num):
        i_batch = [X[i*batch_size:(i+1)*batch_size]] if y is None else [X[i*batch_size:(i+1)*batch_size], y[i*batch_size:(i+1)*batch_size]]
        batched_data.append(i_batch)
    if batch_num * batch_size != len(X):
        last_batch = [X[batch_num*batch_size:]] if y is None else [X[batch_num*batch_size:], y[batch_num*batch_size:]]
        batched_data.append(last_batch)
    return batched_data

X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10]])
y = np.array([1, 2, 3, 4, 5])
batch_size = 2
print(batch_iterator(X, y, batch_size))


"""
Test Case 1: Accepted
Input:
print(batch_iterator(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), np.array([1, 2, 3, 4, 5]), batch_size=2))
Output:
[[array([[1, 2],
       [3, 4]]), array([1, 2])], [array([[5, 6],
       [7, 8]]), array([3, 4])], [array([[ 9, 10]]), array([5])]]
Expected:
[[[[1, 2], [3, 4]], [1, 2]], [[[5, 6], [7, 8]], [3, 4]], [[[9, 10]], [5]]]
Test Case 2: Accepted
Input:
print(batch_iterator(np.array([[1, 1], [2, 2], [3, 3], [4, 4]]), batch_size=3))
Output:
[[array([[1, 1],
       [2, 2],
       [3, 3]])], [array([[4, 4]])]]
Expected:
[[[1, 1], [2, 2], [3, 3]], [[4, 4]]]

import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    n_samples = X.shape[0]
    batches = []
    for i in np.arange(0, n_samples, batch_size):
        begin, end = i, min(i+batch_size, n_samples)
        if y is not None:
            batches.append([X[begin:end], y[begin:end]])
        else:
            batches.append( X[begin:end])
    return batches
    
"""

