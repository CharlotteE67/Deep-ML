# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2025/1/3 12:55
"""
Adam Optimizer
Implement the Adam optimizer update step function. Your function should take the current parameter value, gradient,
and moving averages as inputs, and return the updated parameter value and new moving averages.
The function should also handle scalar and array inputs and include bias correction for the moving averages.

Example:
Input:
parameter = 1.0, grad = 0.1, m = 0.0, v = 0.0, t = 1
Output:
(0.999, 0.01, 0.0001)
Reasoning:
The Adam optimizer computes updated values for the parameter, first moment (m), and second moment (v) using bias-corrected estimates of gradients.
With input values parameter=1.0, grad=0.1, m=0.0, v=0.0, and t=1, the updated parameter becomes 0.999.
"""
import numpy as np


def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
	Update parameters using the Adam optimizer.
	Adjusts the learning rate based on the moving averages of the gradient and squared gradient.
	:param parameter: Current parameter value
	:param grad: Current gradient
	:param m: First moment estimate
	:param v: Second moment estimate
	:param t: Current timestep
	:param learning_rate: Learning rate (default=0.001)
	:param beta1: First moment decay rate (default=0.9)
	:param beta2: Second moment decay rate (default=0.999)
	:param epsilon: Small constant for numerical stability (default=1e-8)
	:return: tuple: (updated_parameter, updated_m, updated_v)
	"""
    # Your code here
    for _ in range(t):
        m = beta1*m + (1-beta1)*grad
        v = beta2*v + (1-beta2)*grad**2

        m_bias = m / (1-beta1)
        v_bias = v / (1-beta2)
        parameter = parameter - learning_rate * (m_bias)/(v_bias**0.5 + epsilon)
    return np.round(parameter, 5), np.round(m, 5), np.round(v, 5)

print(adam_optimizer(parameter = 1.0, grad = 0.1, m = 0.0, v = 0.0, t = 1))


"""
Test Case
print(adam_optimizer(1.0, 0.1, 0.0, 0.0, 1))

Expected Output
(0.999, 0.01, 0.0001)

Actual Output
(0.999, 0.01, 1e-05)

Status
Passed

Test Case
print(adam_optimizer(np.array([1.0, 2.0]), np.array([0.1, 0.2]), np.zeros(2), np.zeros(2), 1))

Expected Output
(array([0.999, 1.999]), array([0.01, 0.02]), array([1.e-05, 4.e-05]))

Actual Output
(array([0.999, 1.999]), array([0.01, 0.02]), array([1.e-05, 4.e-05]))

Status
Passed

import numpy as np

def adam_optimizer(parameter, grad, m, v, t, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
    # Update biased first moment estimate
    m = beta1 * m + (1 - beta1) * grad

    # Update biased second raw moment estimate
    v = beta2 * v + (1 - beta2) * (grad**2)

    # Compute bias-corrected first moment estimate
    m_hat = m / (1 - beta1**t)

    # Compute bias-corrected second raw moment estimate
    v_hat = v / (1 - beta2**t)

    # Update parameters
    update = learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    parameter = parameter - update

    return np.round(parameter,5), np.round(m,5), np.round(v,5)
"""