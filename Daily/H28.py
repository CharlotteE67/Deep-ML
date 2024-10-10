# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/10 13:59
"""
SVD of a 2x2 Matrix using eigen values & vectors
Given a 2x2 matrix, write a Python function to compute its Singular Value Decomposition (SVD). The function should return the matrices U, S, and V such that A = U * S * V, use the method described in this post https://metamerist.blogspot.com/2006/10/linear-algebra-for-graphics-geeks-svd.html
Example
Example:
    A = [[-10, 8],
         [10, -1]]
    output:(array([[  0.8, -0.6], [-0.6, -0.8]]),
    array([15.65247584,  4.47213595]),
    array([[ -0.89442719,  0.4472136], [ -0.4472136 , -0.89442719]]))

"""
import numpy as np


def svd_2x2(A: np.ndarray) -> tuple:
    # Your code here
    s, v, d = np.linalg.svd(A)
    return s, v, d

A = [[-10, 8],
         [10, -1]]
print(svd_2x2(A))

"""
Test Case 1: Accepted
Input:
U,s,V = svd_2x2(np.array([[-10, 8], [10, -1]]))
result = U @ np.diag(s) @ V
print(result)
Output:
[[-10.   8.]
 [ 10.  -1.]]
Expected:
[[-10, 8], [10, -1]]
Test Case 2: Accepted
Input:
U,s,V = svd_2x2(np.array([[1, 2], [3, 4]]))
result = U @ np.diag(s) @ V
print(result)
Output:
[[1. 2.]
 [3. 4.]]
Expected:
[[1, 2], [3, 4]]

Understanding Singular Value Decomposition (SVD)
Singular Value Decomposition (SVD) is a method in linear algebra for decomposing a matrix into three other matrices.
For a given matrix A, SVD is represented as:

A = U*S*V^T

Here's a step-by-step method to calculate the SVD of a 2x2 matrix by hand:

Calculate 
 and 
: Compute the product of the matrix with its transpose and the transpose of the matrix with itself. These matrices share the same eigenvalues.
Find the Eigenvalues: To find the eigenvalues of a 2x2 matrix, solve the characteristic equation 
. This results in a quadratic equation.
Compute the Singular Values: The singular values, which form the diagonal elements of the matrix S, are the square roots of the eigenvalues.
Calculate the Eigenvectors: For each eigenvalue, solve the equation 
 to find the corresponding eigenvector. Normalize these eigenvectors to form the columns of U and V.
Form the Matrices U, S, and V: Combine the singular values and eigenvectors to construct the matrices U, S, and V such that 
.
This method involves solving quadratic equations to find eigenvalues and eigenvectors and normalizing these vectors to unit length.

Resources: Linear Algebra for Graphics Geeks (SVD-IX) by METAMERIST Robust algorithm for 2×2 SVD

import numpy as np

def svd_2x2(A: np.ndarray) -> tuple:
    y1, x1 = (A[1, 0] + A[0, 1]), (A[0, 0] - A[1, 1])
    y2, x2 = (A[1, 0] - A[0, 1]), (A[0, 0] + A[1, 1])

    h1 = np.sqrt(y1**2 + x1**2)
    h2 = np.sqrt(y2**2 + x2**2)

    t1 = x1 / h1
    t2 = x2 / h2

    cc = np.sqrt((1.0 + t1) * (1.0 + t2))
    ss = np.sqrt((1.0 - t1) * (1.0 - t2))
    cs = np.sqrt((1.0 + t1) * (1.0 - t2))
    sc = np.sqrt((1.0 - t1) * (1.0 + t2))

    c1, s1 = (cc - ss) / 2.0, (sc + cs) / 2.0
    U = np.array([[-c1, -s1], [-s1, c1]])

    s = np.array([(h1 + h2) / 2.0, abs(h1 - h2) / 2.0])

    V = np.diag(1.0 / s) @ U.T @ A

    return U, s, V
    
"""