# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/16 14:28
"""Principal Component Analysis (PCA) Implementation (medium) Write a Python function that performs Principal
Component Analysis (PCA) from scratch. The function should take a 2D NumPy array as input, where each row represents
a data sample and each column represents a feature. The function should standardize the dataset, compute the
covariance matrix, find the eigenvalues and eigenvectors, and return the principal components (the eigenvectors
corresponding to the largest eigenvalues). The function should also take an integer k as input, representing the
number of principal components to return.
Example
Example:
    input: data = np.array([[1, 2], [3, 4], [5, 6]]), k = 1
    output:  [[0.7071], [0.7071]]
reasoning: After standardizing the data and computing the covariance matrix,
the eigenvalues and eigenvectors are calculated. The largest eigenvalue's corresponding eigenvector is returned as
the principal component, rounded to four decimal places.
"""
from __future__ import annotations

import numpy as np


def pca(data: np.ndarray, k: int) -> list[list[int | float]]:
    # Your code here
    std_data = (data - data.mean(axis=0)) / data.std(axis=0)
    # print(std_data)
    cov_matrix = np.cov(std_data, rowvar=False)
    # print(cov_matrix)
    eigen_values, eigen_vectors = np.linalg.eig(cov_matrix)
    # print(eigen_vectors, eigen_values)
    pc_idx = np.argsort(eigen_values)[::-1]
    principal_components = eigen_vectors[:, pc_idx]
    principal_components = principal_components[:, :k]
    # print(principal_components)
    return np.round(principal_components, 4)


data = np.array([[1, 2], [3, 4], [5, 6]])
k = 1
print(pca(data, k))

print(pca(np.array([[4,2,1],[5,6,7],[9,12,1],[4,6,7]]),2))



"""
Test Case 1: Accepted
Input:
print(pca(np.array([[4,2,1],[5,6,7],[9,12,1],[4,6,7]]),2))
Output:
[[ 0.6855  0.0776]
 [ 0.6202  0.4586]
 [-0.3814  0.8853]]
Expected:
[[0.6855, 0.0776], [0.6202, 0.4586], [-0.3814, 0.8853]]
Test Case 2: Accepted
Input:
print(pca(np.array([[1, 2], [3, 4], [5, 6]]), k = 1))
Output:
[[0.7071]
 [0.7071]]
Expected:
[[0.7071], [0.7071]]


import numpy as np

def pca(data, k):
    # Standardize the data
    data_standardized = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
    
    # Compute the covariance matrix
    covariance_matrix = np.cov(data_standardized, rowvar=False)
    
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(covariance_matrix)
    
    # Sort the eigenvectors by decreasing eigenvalues
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues_sorted = eigenvalues[idx]
    eigenvectors_sorted = eigenvectors[:,idx]
    
    # Select the top k eigenvectors (principal components)
    principal_components = eigenvectors_sorted[:, :k]
    
    return np.round(principal_components, 4).tolist()
"""