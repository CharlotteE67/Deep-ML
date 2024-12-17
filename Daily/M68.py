# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/12/17 11:04
"""
Find the Image of a Matrix Using Row Echelon Form
Description:
Task: Compute the Column Space of a Matrix
In this task, you are required to implement a function matrix_image(A) that calculates the column space of a given matrix A.
The column space, also known as the image or span, consists of all linear combinations of the columns of A.
To find this, you'll use concepts from linear algebra, focusing on identifying independent columns that span the matrix's image.
Your task: Implement the function matrix_image(A) to return the basis vectors that span the column space of A.
These vectors should be extracted from the original matrix and correspond to the independent columns.

Example:
Input:
matrix = np.array([
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
])
print(matrix_image(matrix))
Output:
# [[1, 2],
#  [4, 5],
#  [7, 8]]
Reasoning:
The column space of the matrix is spanned by the independent columns [1, 2], [4, 5], and [7, 8].
These columns form the basis vectors that represent the image of the matrix.

Test Cases:
Test:

import numpy as np
matrix = np.array([[1, 0], [0, 1]])
print(matrix_image(matrix))
Expected Output:
[[1, 0], [0, 1]]
Test:

import numpy as np
matrix = np.array([[1, 2], [2, 4]])
print(matrix_image(matrix))
Expected Output:
[[1], [2]]

"""
import numpy as np

def rref(A):
    # Convert to float for division operations
    A = A.astype(np.float32)
    n, m = A.shape

    for i in range(n):
        if A[i, i] == 0:
            nonzero_current_row = np.nonzero(A[i:, i])[0] + i
            if len(nonzero_current_row) == 0:
                continue
            A[[i, nonzero_current_row[0]]] = A[[nonzero_current_row[0], i]]

        A[i] = A[i] / A[i, i]

        for j in range(n):
            if i != j:
                A[j] -= A[i] * A[j, i]
    return A

def find_pivot_columns(A):
    n, m = A.shape
    pivot_columns = []
    for i in range(n):
        nonzero = np.nonzero(A[i, :])[0]
        if len(nonzero) != 0:
            pivot_columns.append(nonzero[0])
    return pivot_columns

def matrix_image(A):
    # Find the RREF of the matrix
    Arref = rref(A)
    # Find the pivot columns
    pivot_columns = find_pivot_columns(Arref)
    # Extract the pivot columns from the original matrix
    image_basis = A[:, pivot_columns]
    return image_basis


matrix = np.array([[1, 0], [0, 1]])
print(matrix_image(matrix))

matrix = np.array([[1, 2], [2, 4]])
print(matrix_image(matrix))