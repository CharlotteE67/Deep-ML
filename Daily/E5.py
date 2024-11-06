# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/11/6 12:38
"""
Scalar Multiplication of a Matrix (easy)
Write a Python function that multiplies a matrix by a scalar and returns the result.
Example
Example:
        input: matrix = [[1, 2], [3, 4]], scalar = 2
        output: [[2, 4], [6, 8]]
        reasoning: Each element of the matrix is multiplied by the scalar.
"""
from __future__ import annotations


def scalar_multiply(matrix: list[list[int | float]], scalar: int | float) -> list[list[int | float]]:
    result = []
    for i in range(len(matrix)):
        i_result = []
        for j in range(len(matrix[i])):
            i_result.append(scalar * matrix[i][j])
        result.append(i_result)
    return result


print(scalar_multiply(matrix = [[1, 2], [3, 4]], scalar = 2))

"""
Test Case 1: Accepted
Input:
print(scalar_multiply([[1,2],[3,4]], 2))
Output:
[[2, 4], [6, 8]]
Expected:
[[2, 4], [6, 8]]
Test Case 2: Accepted
Input:
print(scalar_multiply([[0,-1],[1,0]], -1))
Output:
[[0, 1], [-1, 0]]
Expected:
[[0, 1], [-1, 0]]

def scalar_multiply(matrix: list[list[int|float]], scalar: int|float) -> list[list[int|float]]:
    return [[element * scalar for element in row] for row in matrix]
"""