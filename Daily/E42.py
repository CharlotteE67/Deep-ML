# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/8 13:02
"""
Implement ReLU Activation Function
Write a Python function `relu` that implements the Rectified Linear Unit (ReLU) activation function. The function should take a single float as input and return the value after applying the ReLU function. The ReLU function returns the input if it's greater than 0, otherwise, it returns 0.
Example
Example:
print(relu(0))
# Output: 0

print(relu(1))
# Output: 1

print(relu(-1))
# Output: 0
"""


def relu(z: float) -> float:
    # Your code here
    return z if z > 0 else 0


print(relu(0))
# Output: 0

print(relu(1))
# Output: 1

print(relu(-1))
# Output: 0

"""
Test Case 1: Accepted
Input:
print(relu(0))
Output:
0
Expected:
0
Test Case 2: Accepted
Input:
print(relu(1))
Output:
1
Expected:
1
Test Case 3: Accepted
Input:
print(relu(-1))
Output:
0
Expected:
0
"""