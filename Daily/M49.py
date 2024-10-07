# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/7 15:35
"""
Implement Adam Optimization Algorithm
Implement the Adam (Adaptive Moment Estimation) optimization algorithm in Python. Adam is an optimization algorithm that adapts the learning rate for each parameter. Your task is to write a function `adam_optimizer` that updates the parameters of a given function using the Adam algorithm.

The function should take the following parameters:

- `f`: The objective function to be optimized
- `grad`: A function that computes the gradient of `f`
- `x0`: Initial parameter values
- `learning_rate`: The step size (default: 0.001)
- `beta1`: Exponential decay rate for the first moment estimates (default: 0.9)
- `beta2`: Exponential decay rate for the second moment estimates (default: 0.999)
- `epsilon`: A small constant for numerical stability (default: 1e-8)
- `num_iterations`: Number of iterations to run the optimizer (default: 1000)
The function should return the optimized parameters.

Example
Example:
import numpy as np

def objective_function(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return np.array([2*x[0], 2*x[1]])

x0 = np.array([1.0, 1.0])
x_opt = adam_optimizer(objective_function, gradient, x0)

print("Optimized parameters:", x_opt)

# Expected Output:
# Optimized parameters: [0.99000325 0.99000325]
"""

import numpy as np


def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    # Your code here
    # global x_t
    m_tm1 = v_tm1 = 0
    x_tm1 = x0
    x_t = None
    for t in range(1, num_iterations + 1):
        g_t = grad(x_tm1)
        m_t = beta1 * m_tm1 + (1 - beta1) * g_t
        v_t = beta2 * v_tm1 + (1 - beta2) * g_t ** 2
        m_t_head = m_t / (1 - beta1 ** t)
        v_t_head = v_t / (1 - beta2 ** t)
        update = learning_rate * m_t_head / (v_t_head ** 0.5 + epsilon)
        x_t = x_tm1 - update
        x_tm1, m_tm1, v_tm1 = x_t, m_t, v_t

    return x_t


def objective_function(x):
    return x[0] ** 2 + x[1] ** 2


def gradient(x):
    return np.array([2 * x[0], 2 * x[1]])


x0 = np.array([1.0, 1.0])
x_opt = adam_optimizer(objective_function, gradient, x0)

print("Optimized parameters:", x_opt)


"""
import numpy as np

def adam_optimizer(f, grad, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, num_iterations=10):
    x = x0
    m = np.zeros_like(x)
    v = np.zeros_like(x)

    for t in range(1, num_iterations + 1):
        g = grad(x)
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g**2
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        x = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

    return x

Test Case 1: Accepted
Input:
import numpy as np

def objective_function(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return np.array([2*x[0], 2*x[1]])

x0 = np.array([1.0, 1.0])
x_opt = adam_optimizer(objective_function, gradient, x0)

print(x_opt)
Output:
[0.99000325 0.99000325]
Expected:
[0.99000325 0.99000325]
Test Case 2: Accepted
Input:
import numpy as np

def objective_function(x):
    return x[0]**2 + x[1]**2

def gradient(x):
    return np.array([2*x[0], 2*x[1]])

x0 = np.array([0.2, 12.3])
x_opt = adam_optimizer(objective_function, gradient, x0)

print(x_opt)
Output:
[ 0.19001678 12.29000026]
Expected:
[ 0.19001678 12.29000026]
"""