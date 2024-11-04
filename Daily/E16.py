# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/11/4 11:30
"""
Feature Scaling Implementation (easy)
Write a Python function that performs feature scaling on a dataset using both standardization and min-max normalization.
The function should take a 2D NumPy array as input, where each row represents a data sample and each column represents a feature.
It should return two 2D NumPy arrays: one scaled by standardization and one by min-max normalization.
Make sure all results are rounded to the nearest 4th decimal.
Example
Example:
        input: data = np.array([[1, 2], [3, 4], [5, 6]])
        output: ([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])
        reasoning: Standardization rescales the feature to have a mean of 0 and a standard deviation of 1.
        Min-max normalization rescales the feature to a range of [0, 1], where the minimum feature value
        maps to 0 and the maximum to 1.

"""
import numpy as np


def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
    # Your code here
    data_mean = np.mean(data, axis=0)
    data_std = np.std(data, axis=0)
    data_max, data_min = np.max(data, axis=0), np.min(data, axis=0)
    standardized_data = np.array([(data[i] - data_mean) / data_std for i in range(len(data))])
    normalized_data = np.array([(data[i] - data_min) / (data_max - data_min) for i in range(len(data))])
    standardized_data, normalized_data = np.round(standardized_data, 4), np.round(normalized_data, 4)
    return standardized_data, normalized_data


data = np.array([[1, 2], [3, 4], [5, 6]])
print(feature_scaling(data))

"""
Test Case 1: Accepted
Input:
print(feature_scaling(np.array([[1, 2], [3, 4], [5, 6]])))
Output:
(array([[-1.2247, -1.2247],
       [ 0.    ,  0.    ],
       [ 1.2247,  1.2247]]), array([[0. , 0. ],
       [0.5, 0.5],
       [1. , 1. ]]))
Expected:
([[-1.2247, -1.2247], [0.0, 0.0], [1.2247, 1.2247]], [[0.0, 0.0], [0.5, 0.5], [1.0, 1.0]])


import numpy as np

def feature_scaling(data):
    # Standardization
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)
    standardized_data = (data - mean) / std
    
    # Min-Max Normalization
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    normalized_data = (data - min_val) / (max_val - min_val)
    
    return np.round(standardized_data,4).tolist(), np.round(normalized_data,4).tolist()
"""