# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/11/24 16:57
"""
Reshape Matrix (easy)
Write a Python function that reshapes a given matrix into a specified shape.
Example
Example:
        input: a = [[1,2,3,4],[5,6,7,8]], new_shape = (4, 2)
        output: [[1, 2], [3, 4], [5, 6], [7, 8]]
        reasoning: The given matrix is reshaped from 2x4 to 4x2.
"""
from __future__ import annotations

import numpy as np


def reshape_matrix(a: list[list[int | float]], new_shape: tuple[int, int]) -> list[list[int | float]]:
    # Write your code here and return a python list after reshaping by using numpy's tolist() method
    reshaped_matrix = np.reshape(np.array(a), new_shape).tolist()
    return reshaped_matrix

a = [[1,2,3,4],[5,6,7,8]]
new_shape = (4, 2)
print(reshape_matrix(a, new_shape))

"""
Test Case 1: Accepted
Input:
print(reshape_matrix([[1,2,3,4],[5,6,7,8]], (4, 2)))
Output:
[[1, 2], [3, 4], [5, 6], [7, 8]]
Expected:
[[1, 2], [3, 4], [5, 6], [7, 8]]
Test Case 2: Accepted
Input:
print(reshape_matrix([[1,2,3],[4,5,6]], (3, 2)))
Output:
[[1, 2], [3, 4], [5, 6]]
Expected:
[[1, 2], [3, 4], [5, 6]]
Test Case 3: Accepted
Input:
print(reshape_matrix([[1,2,3,4],[5,6,7,8]], (2, 4)))
Output:
[[1, 2, 3, 4], [5, 6, 7, 8]]
Expected:
[[1, 2, 3, 4], [5, 6, 7, 8]]

import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int|float]) -> list[list[int|float]]:
    return np.array(a).reshape(new_shape).tolist()
"""