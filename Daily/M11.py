# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/30 13:31
"""
Solve Linear Equations using Jacobi Method (medium)
Write a Python function that uses the Jacobi method to solve a
system of linear equations given by Ax = b. The function should iterate n times, rounding each intermediate solution
to four decimal places, and return the approximate solution x.
Example
Example:
    input: A = [[5, -2, 3], [-3, 9, 1],[2, -1, -7]], b = [-1, 2, 3], n=2
    output: [0.146, 0.2032, -0.5175]
    reasoning: The Jacobi method iteratively solves
each equation for x[i] using the formula x[i] = (1/a_ii) * (b[i] - sum(a_ij * x[j] for j != i)), where a_ii is the
diagonal element of A and a_ij are the off-diagonal elements.
"""

import numpy as np


def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    x = np.zeros(len(b))
    next_x = np.zeros(len(b))
    for _ in range(n):
        for i in range(len(x)):
            # print([(A[i][j], j) for j in range(len(x)) if j != i])
            next_x[i] = round(1/A[i][i] * (b[i] - sum([A[i][j] * x[j] for j in range(len(x)) if j != i])), 4)
            # x[j] = round(1 / A[j][j] * (b[j] - sum([A[j][k] * x[k] for k in range(len(x)) if k != j])), 4)
        x = next_x.copy()
        next_x = np.zeros(len(b))
    return np.round(x, 4).tolist()

# def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
#     d_a = np.diag(A)
#     nda = A - np.diag(d_a)
#     x = np.zeros(len(b))
#     x_hold = np.zeros(len(b))
#     for _ in range(n):
#         for i in range(len(A)):
#             x_hold[i] = (1/d_a[i]) * (b[i] - sum(nda[i]*x))
#         x = x_hold.copy()
#     return np.round(x,4).tolist()

A = [[5, -2, 3], [-3, 9, 1], [2, -1, -7]]
b = [-1, 2, 3]
n = 2
print(solve_jacobi(np.array(A), np.array(b), n))

"""
Test Case 1: Accepted
Input:
print(solve_jacobi(np.array([[5, -2, 3], [-3, 9, 1], [2, -1, -7]]), np.array([-1, 2, 3]),2))
Output:
[0.146, 0.2032, -0.5175]
Expected:
[0.146, 0.2032, -0.5175]
Test Case 2: Accepted
Input:
print(solve_jacobi(np.array([[4, 1, 2], [1, 5, 1], [2, 1, 3]]), np.array([4, 6, 7]),5))
Output:
[-0.0805, 0.9324, 2.4422]
Expected:
[-0.0806, 0.9324, 2.4422]
Test Case 3: Accepted
Input:
print(solve_jacobi(np.array([[4,2,-2],[1,-3,-1],[3,-1,4]]), np.array([0,7,5]),3))
Output:
[1.7084, -1.9584, -0.7812]
Expected:
[1.7083, -1.9583, -0.7812]

import numpy as np

def solve_jacobi(A: np.ndarray, b: np.ndarray, n: int) -> list:
    d_a = np.diag(A)
    nda = A - np.diag(d_a)
    x = np.zeros(len(b))
    x_hold = np.zeros(len(b))
    for _ in range(n):
        for i in range(len(A)):
            x_hold[i] = (1/d_a[i]) * (b[i] - sum(nda[i]*x))
        x = x_hold.copy()
    return np.round(x,4).tolist()
"""
