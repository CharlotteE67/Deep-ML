# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/29 12:10
"""
Linear Kernel Function
Write a Python function `kernel_function` that computes the linear kernel between two input
vectors `x1` and `x2`. The linear kernel is defined as the dot product (inner product) of two vectors. Example
Example:
import numpy as np

x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])

result = kernel_function(x1, x2)
print(result)
# Expected Output: 32
"""
import numpy as np


def kernel_function(x1, x2):
    # Your code here
    return np.dot(x1, x2)


import numpy as np

x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])

result = kernel_function(x1, x2)
print(result)

"""
Test Case 1: Accepted
Input:
import numpy as np
x1 = np.array([1, 2, 3])
x2 = np.array([4, 5, 6])
result = kernel_function(x1, x2)
print(result)
Output:
32
Expected:
32
Test Case 2: Accepted
Input:
import numpy as np
x1 = np.array([0, 1, 2])
x2 = np.array([3, 4, 5])
result = kernel_function(x1, x2)
print(result)
Output:
14
Expected:
14

import numpy as np

def kernel_function(x1, x2):
    return np.inner(x1, x2)

"""