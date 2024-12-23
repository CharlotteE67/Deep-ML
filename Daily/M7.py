# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/12/23 12:47
"""
Matrix Transformation
Description:
Write a Python function that transforms a given matrix A using the operationT^{−1}AS, where T and S are invertible matrices.
The function should first validate if the matrices T and S are invertible, and then perform the transformation.
In cases where there is no solution return -1

Example:
Input:
A = [[1, 2], [3, 4]], T = [[2, 0], [0, 2]], S = [[1, 1], [0, 1]]
Output:
[[0.5,1.5],[1.5,3.5]]
Reasoning:
The matrices T and S are used to transform matrix A by computingT^{−1}AS.

Test Cases:
Test:
print(transform_matrix([[1, 2], [3, 4]], [[2, 0], [0, 2]], [[1, 1], [0, 1]]))
Expected Output:
[[0.5, 1.5], [1.5, 3.5]]
Test:
print(transform_matrix([[1, 0], [0, 1]], [[1, 2], [3, 4]], [[2, 0], [0, 2]]))
Expected Output:
[[-4.0, 2.0], [3.0, -1.0]]
"""
from __future__ import annotations

import numpy as np


def transform_matrix(A: list[list[int | float]], T: list[list[int | float]], S: list[list[int | float]]) -> list[
    list[int | float]]:
    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
        return -1
    # print(np.linalg.inv(T))
    transformed_matrix = np.dot(np.dot(np.linalg.inv(T), A), S)
    return transformed_matrix

print(transform_matrix(A = [[1, 2], [3, 4]], T = [[2, 0], [0, 2]], S = [[1, 1], [0, 1]]))


"""
Test Case
print(transform_matrix([[1, 2], [3, 4]], [[2, 0], [0, 2]], [[1, 1], [0, 1]]))

Expected Output
[[0.5, 1.5], [1.5, 3.5]]

Actual Output
[[0.5 1.5] [1.5 3.5]]

Status
Passed

Test Case
print(transform_matrix([[1, 0], [0, 1]], [[1, 2], [3, 4]], [[2, 0], [0, 2]]))

Expected Output
[[-4.0, 2.0], [3.0, -1.0]]

Actual Output
[[-4. 2.] [ 3. -1.]]

Status
Passed


import numpy as np

def transform_matrix(A: list[list[int|float]], T: list[list[int|float]], S: list[list[int|float]]) -> list[list[int|float]]:
    # Convert to numpy arrays for easier manipulation
    A = np.array(A, dtype=float)
    T = np.array(T, dtype=float)
    S = np.array(S, dtype=float)
    
    # Check if the matrices T and S are invertible
    if np.linalg.det(T) == 0 or np.linalg.det(S) == 0:
        # raise ValueError("The matrices T and/or S are not invertible.")
        return -1
    
    # Compute the inverse of T
    T_inv = np.linalg.inv(T)

    # Perform the matrix transformation; use @ for better readability
    transformed_matrix = np.round(T_inv @ A @ S, 3)
    
    return transformed_matrix.tolist()
"""
