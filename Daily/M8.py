# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/9/27 20:52
"""
https://www.deep-ml.com/problem/Calculate%202x2%20Matrix%20Inverse
Write a Python function that calculates the inverse of a 2x2 matrix. Return 'None' if the matrix is not invertible.
Calculate 2x2 Matrix Inverse (medium)
Example
Example:
        input: matrix = [[4, 7], [2, 6]]
        output: [[0.6, -0.7], [-0.2, 0.4]]
        reasoning: The inverse of a 2x2 matrix [a, b], [c, d] is given by (1/(ad-bc)) * [d, -b], [-c, a], provided ad-bc is not zero.
"""
def inverse_2x2(matrix: list[list[float]]) -> list[list[float]]:
    a, b, c, d = matrix[0][0], matrix[0][1], matrix[1][0], matrix[1][1]
    factor = 1 / (a*d - b*c)
    inverse = [[factor*d, -factor*b], [-factor*c, factor*a]]
    return inverse


"""
Test Case 1: Accepted
Input:
print(inverse_2x2([[4, 7], [2, 6]]))
Output:
[[0.6000000000000001, -0.7000000000000001], [-0.2, 0.4]]
Expected:
[[0.6, -0.7], [-0.2, 0.4]]

Test Case 2: Accepted
Input:
print(inverse_2x2([[2, 1], [6, 2]]))
Output:
[[-1.0, 0.5], [3.0, -1.0]]
Expected:
[[-1.0, 0.5], [3.0, -1.0]]
"""