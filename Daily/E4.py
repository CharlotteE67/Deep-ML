# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/11/5 15:15
"""
Calculate Mean by Row or Column (easy)
Write a Python function that calculates the mean of a matrix either by row or by column, based on a given mode.
The function should take a matrix (list of lists) and a mode ('row' or 'column') as input and return a list of means according to the specified mode.
Example
Example1:
        input: matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'column'
        output: [4.0, 5.0, 6.0]
        reasoning: Calculating the mean of each column results in [(1+4+7)/3, (2+5+8)/3, (3+6+9)/3].

        Example 2:
        input: matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'row'
        output: [2.0, 5.0, 8.0]
        reasoning: Calculating the mean of each row results in [(1+2+3)/3, (4+5+6)/3, (7+8+9)/3].
"""
from typing import List


def calculate_matrix_mean(matrix: List[List[float]], mode: str) -> List[float]:
    means = []
    row, col = len(matrix), len(matrix[0])
    if mode == "column":
        means = [0] * col
        for i in range(row):
            for j in range(col):
                means[j] += matrix[i][j]
        means = [item/row for item in means]
    elif mode == "row":
        means = [0] * row
        for i in range(row):
            for j in range(col):
                means[i] += matrix[i][j]
        means = [item/col for item in means]
    return means

print(calculate_matrix_mean(matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'column'))
print(calculate_matrix_mean(matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]], mode = 'row'))


"""
Test Case 1: Accepted
Input:
print(calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'column'))
Output:
[4.0, 5.0, 6.0]
Expected:
[4.0, 5.0, 6.0]
Test Case 2: Accepted
Input:
print(calculate_matrix_mean([[1, 2, 3], [4, 5, 6], [7, 8, 9]], 'row'))
Output:
[2.0, 5.0, 8.0]
Expected:
[2.0, 5.0, 8.0]

def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    if mode == 'column':
        return [sum(col) / len(matrix) for col in zip(*matrix)]
    elif mode == 'row':
        return [sum(row) / len(row) for row in matrix]
    else:
        raise ValueError("Mode must be 'row' or 'column'")
"""