# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/12/25 11:30
"""
Calculate Correlation Matrix
Description:
Write a Python function to calculate the correlation matrix for a given dataset.
The function should take in a 2D numpy array X and an optional 2D numpy array Y.
If Y is not provided, the function should calculate the correlation matrix of X with itself.
It should return the correlation matrix as a 2D numpy array.

Example:
Input:
X = np.array([[1, 2],
                  [3, 4],
                  [5, 6]])
    output = calculate_correlation_matrix(X)
    print(output)
Output:
# [[1. 1.]
    #  [1. 1.]]
Reasoning:
The function calculates the correlation matrix for the dataset X.
In this example, the correlation between the two features is 1, indicating a perfect linear relationship.

Test Cases:
Test:
print(calculate_correlation_matrix(np.array([[1, 2], [3, 4], [5, 6]])))
Expected Output:
[[1., 1.], [1., 1.]]
Test:
print(calculate_correlation_matrix(np.array([[1, 2, 3], [7, 15, 6], [7, 8, 9]])))
Expected Output:
[[1.,0.84298868, 0.8660254 ],[0.84298868, 1., 0.46108397],[0.8660254,  0.46108397, 1.]]

"""
import numpy as np


def calculate_correlation_matrix(X, Y=None):
    # Your code here
    # if Y:
    #     return np.corrcoef(X.T, Y.T)
    #
    # return np.corrcoef(X.T)

    # Helper function to calculate standard deviation
    def calculate_std_dev(A):
        return np.sqrt(np.mean((A - A.mean(0)) ** 2, axis=0))

    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    # Calculate the covariance matrix
    covariance = (1 / n_samples) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    # Calculate the standard deviations
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)
    # Calculate the correlation matrix
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype=float)


X = np.array([[1, 2],
              [3, 4],
              [5, 6]])
output = calculate_correlation_matrix(X)
print(output)

print(calculate_correlation_matrix(np.array([[1, 2, 3], [7, 15, 6], [7, 8, 9]])))


"""
Understanding Correlation Matrix
A correlation matrix is a table showing the correlation coefficients between variables. 
Each cell in the table shows the correlation between two variables, with values ranging from -1 to 1. 
These values indicate the strength and direction of the linear relationship between the variables.

Mathematical Definition
The correlation coefficient between two variables ( X ) and ( Y ) is given by:

corr(X,Y)=cov(X,Y)/(σ_X*σ_Y)
Where:
( \text{cov}(X, Y) ) is the covariance between ( X ) and ( Y ).
( \sigma_X ) and ( \sigma_Y ) are the standard deviations of ( X ) and ( Y ), respectively.
Problem Overview
In this problem, you will write a function to calculate the correlation matrix for a given dataset. T
he function will take in a 2D numpy array ( X ) and an optional 2D numpy array ( Y ). 
If ( Y ) is not provided, the function will calculate the correlation matrix of ( X ) with itself.

import numpy as np

def calculate_correlation_matrix(X, Y=None):
    # Helper function to calculate standard deviation
    def calculate_std_dev(A):
        return np.sqrt(np.mean((A - A.mean(0))**2, axis=0))
    
    if Y is None:
        Y = X
    n_samples = np.shape(X)[0]
    # Calculate the covariance matrix
    covariance = (1 / n_samples) * (X - X.mean(0)).T.dot(Y - Y.mean(0))
    # Calculate the standard deviations
    std_dev_X = np.expand_dims(calculate_std_dev(X), 1)
    std_dev_y = np.expand_dims(calculate_std_dev(Y), 1)
    # Calculate the correlation matrix
    correlation_matrix = np.divide(covariance, std_dev_X.dot(std_dev_y.T))

    return np.array(correlation_matrix, dtype=float)
    
"""