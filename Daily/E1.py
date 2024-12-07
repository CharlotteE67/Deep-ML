# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/12/7 12:35
"""
Matrix times Vector (easy)
Write a Python function that takes the dot product of a matrix and a vector. return -1 if the matrix could not be dotted with the vector
Example
Example:
        input: a = [[1,2],[2,4]], b = [1,2]
        output:[5, 10]
        reasoning: 1*1 + 2*2 = 5;
                   1*2+ 2*4 = 10

"""
from __future__ import annotations


def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
    col_dim = list(set([len(item) for item in a]))
    if len(col_dim) != 1 or col_dim[0] != len(b):
        return -1
    c = [sum([a[i][j] * b[j] for j in range(len(a[i]))]) for i in range(len(a))]
    return c

a = [[1,2],[2,4]]
b = [1,2]
print(matrix_dot_vector(a, b))

"""
Test Case 1: Accepted
Input:
print(matrix_dot_vector([[1,2,3],[2,4,5],[6,8,9]],[1,2,3]))
Output:
[14, 25, 49]
Expected:
[14, 25, 49]
Test Case 2: Accepted
Input:
print(matrix_dot_vector([[1,2],[2,4],[6,8],[12,4]],[1,2,3]))
Output:
-1
Expected:
-1
Test Case 3: Accepted
Input:
print(matrix_dot_vector([[1, 2, 3], [2, 4, 6]],[1, 2, 3]))
Output:
[14, 28]
Expected:
[14, 28]

def matrix_dot_vector(a:list[list[int|float]],b:list[int|float])-> list[int|float]:
    if len(a[0]) != len(b):
        return -1
    vals = []
    for i in a:
        hold = 0
        for j in range(len(i)):
            hold+=(i[j] * b[j])
        vals.append(hold)

    return vals
"""