# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/27 16:36
"""Write a Python function that performs k-fold cross-validation data splitting from scratch. The function should
take a dataset (as a 2D NumPy array where each row represents a data sample and each column represents a feature) and
an integer k representing the number of folds. The function should split the dataset into k parts, systematically use
one part as the test set and the remaining as the training set, and return a list where each element is a tuple
containing the training set and test set for each fold.
Example
Example:
    input: data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), k = 5
    output: [[[[3, 4], [5, 6], [7, 8], [9, 10]], [[1, 2]]],
            [[[1, 2], [5, 6], [7, 8], [9, 10]], [[3, 4]]],
            [[[1, 2], [3, 4], [7, 8], [9, 10]], [[5, 6]]],
            [[[1, 2], [3, 4], [5, 6], [9, 10]], [[7, 8]]],
            [[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10]]]]
    reasoning: The dataset is divided into 5 parts, each being used once as a test set while the remaining parts serve as the training set.

"""

import numpy as np


def cross_validation_split(data: np.ndarray, k: int, seed=42) -> list:
    np.random.seed(seed)
    np.random.shuffle(data)

    n, m = data.shape
    sub_size = int(np.ceil(n / k))
    id_s = np.arange(0, n, sub_size)
    print(id_s)
    id_e = id_s + sub_size
    print(id_e)
    if id_e[-1] > n:
        id_e[-1] = n

    return [[np.concatenate([data[: id_s[i]], data[id_e[i]:]], axis=0).tolist(), data[id_s[i]: id_e[i]].tolist()] for i
            in range(k)]


data = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
# np.random.shuffle(data)
# print(data)
k = 5
print(cross_validation_split(data, k))
"""
Understanding k-Fold Cross-Validation Data Splitting
k-Fold cross-validation is a technique used to evaluate the generalizability of a model by dividing the data into `k` folds or subsets. 
Each fold acts as a test set once, with the remaining `k-1` folds serving as the training set. 
This approach ensures that every data point gets used for both training and testing, improving model validation.
Steps in k-Fold Cross-Validation Data Split:
1.Shuffle the dataset randomly. (but not in this case because we test for a unique result)
2.Split the dataset into k groups.
3.Generate Data Splits: For each group, treat that group as the test set and the remaining groups as the training set.
Benefits of this Approach:
- Ensures all data is used for both training and testing. 
- Reduces bias since each data point gets to be in a test set exactly once. 
- Provides a more robust estimate of model performance. 
Implementing this data split function will allow a deeper understanding of how data partitioning affects machine learning models and will provide a foundation for more complex validation techniques.

Test Case 1: Accepted
Input:
print(cross_validation_split(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), 5, 42))
Output:
[[[[9, 10], [5, 6], [1, 2], [7, 8]], [[3, 4]]], [[[3, 4], [5, 6], [1, 2], [7, 8]], [[9, 10]]], [[[3, 4], [9, 10], [1, 2], [7, 8]], [[5, 6]]], [[[3, 4], [9, 10], [5, 6], [7, 8]], [[1, 2]]], [[[3, 4], [9, 10], [5, 6], [1, 2]], [[7, 8]]]]
Expected:
[[[[9, 10], [5, 6], [1, 2], [7, 8]], [[3, 4]]], [[[3, 4], [5, 6], [1, 2], [7, 8]], [[9, 10]]], [[[3, 4], [9, 10], [1, 2], [7, 8]], [[5, 6]]], [[[3, 4], [9, 10], [5, 6], [7, 8]], [[1, 2]]], [[[3, 4], [9, 10], [5, 6], [1, 2]], [[7, 8]]]]
Test Case 2: Accepted
Input:
print(cross_validation_split(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]]), 2, 42))
Output:
[[[[1, 2], [7, 8]], [[3, 4], [9, 10], [5, 6]]], [[[3, 4], [9, 10], [5, 6]], [[1, 2], [7, 8]]]]
Expected:
[[[[1, 2], [7, 8]], [[3, 4], [9, 10], [5, 6]]], [[[3, 4], [9, 10], [5, 6]], [[1, 2], [7, 8]]]]
Test Case 3: Accepted
Input:
print(cross_validation_split(np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]), 3, 42))
Output:
[[[[15, 16], [5, 6], [9, 10], [7, 8], [13, 14]], [[3, 4], [11, 12], [1, 2]]], [[[3, 4], [11, 12], [1, 2], [7, 8], [13, 14]], [[15, 16], [5, 6], [9, 10]]], [[[3, 4], [11, 12], [1, 2], [15, 16], [5, 6], [9, 10]], [[7, 8], [13, 14]]]]
Expected:
[[[[15, 16], [5, 6], [9, 10], [7, 8], [13, 14]], [[3, 4], [11, 12], [1, 2]]], [[[3, 4], [11, 12], [1, 2], [7, 8], [13, 14]], [[15, 16], [5, 6], [9, 10]]], [[[3, 4], [11, 12], [1, 2], [15, 16], [5, 6], [9, 10]], [[7, 8], [13, 14]]]]
"""