"""
Transpose of a Matrix (easy)
Write a Python function that computes the transpose of a given matrix.
Example
Example:
        input: a = [[1,2,3],[4,5,6]]
        output: [[1,4],[2,5],[3,6]]
        reasoning: The transpose of a matrix is obtained by flipping rows and columns.
"""
from __future__ import annotations

def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    b = []
    for i in range(len(a[-1])):
        i_col = []
        for j in range(len(a)):
            i_col.append(a[j][i])
        b.append(i_col)
    return b

a = [[1,2,3],[4,5,6]]
b = transpose_matrix(a)
print(b)

"""
Test Case 1: Accepted
Input:
print(transpose_matrix([[1,2],[3,4],[5,6]]))
Output:
[[1, 3, 5], [2, 4, 6]]
Expected:
[[1, 3, 5], [2, 4, 6]]
Test Case 2: Accepted
Input:
print(transpose_matrix([[1,2,3],[4,5,6]]))
Output:
[[1, 4], [2, 5], [3, 6]]
Expected:
[[1, 4], [2, 5], [3, 6]]
"""