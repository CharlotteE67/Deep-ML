# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/12/31 12:50
"""
Positional Encoding Calculator
Write a Python function to implement the Positional Encoding layer for Transformers.
The function should calculate positional encodings for a sequence length (position) and model dimensionality (d_model) using sine and cosine functions as specified in the Transformer architecture.
The function should return -1 if position is 0, or if d_model is less than or equal to 0. The output should be a numpy array of type float16.

Example:
Input:
position = 2, d_model = 8
Output:
[[[ 0.,0.,0.,0.,1.,1.,1.,1.,]
  [ 0.8413,0.0998,0.01,0.001,0.5405,0.995,1.,1.]]]
Reasoning:
The function computes the positional encoding by calculating sine values for even indices and cosine values for odd indices,
ensuring that the encoding provides the required positional information.
"""
import math

import numpy as np


def pos_encoding(position: int, d_model: int):
    # Your code here
    pos_encoding = np.zeros((position, d_model))
    # pos_encoding.require_grad = False

    position = np.expand_dims(np.arange(0, position), 1)
    div_term = np.exp((np.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)))

    pos_encoding[:, 0::2] = np.sin(position * div_term)  # (position, d_model/2)
    pos_encoding[:, 1::2] = np.cos(position * div_term)  # (position, d_model/2)

    pos_encoding = np.concatenate([pos_encoding[:, 0::2], pos_encoding[:, 1::2]], axis=1)
    pos_encoding = np.expand_dims(pos_encoding, 0)  # (1, position, d_model)
    pos_encoding = np.float16(pos_encoding)
    return pos_encoding

print(pos_encoding(position = 2, d_model = 8))


"""
Test Case
print(pos_encoding(2, 8))

Expected Output
[[[0., 0., 0., 0., 1., 1., 1., 1. ], [0.8413, 0.09985, 0.01, 0.001, 0.5405, 0.995, 1., 1. ]]]

Actual Output
[[[0. 0. 0. 0. 1. 1. 1. 1. ] [0.8413 0.09985 0.01 0.001 0.5405 0.995 1. 1. ]]]

Status
Passed

Test Case
print(pos_encoding(5, 16))

Expected Output
[[[ 0.,0.,0.,0.,0.,0.,0.,0.,1.,1.,1.,1.,1.,1.,1.,1.,], [ 0.8413,0.311,0.09985,0.03162,0.01,0.003162,0.001,0.0003161, 0.5405,0.9502,0.995,0.9995,1.,1.,1.,1.,], [ 0.9092,0.5913,0.1986,0.06323,0.02,0.006325, 0.002001, 0.0006323, -0.4163, 0.8066, 0.98, 0.998, 1., 1.,1.,1.], [ 0.1411,0.8125,0.2954,0.09473,0.03,0.009483, 0.003,0.0009489, -0.9902,0.5825,0.9556,0.9956,0.9995,1.,1.,1.,], [-0.7568,0.9536,0.3894,0.1261,0.04,0.01265,0.004002, 0.001265, -0.6538,0.301,0.9209,0.9922,0.999,1.,1.,1.]]]

Actual Output
[[[ 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 0.000e+00 1.000e+00 1.000e+00 1.000e+00 1.000e+00 1.000e+00 1.000e+00 1.000e+00 1.000e+00] [ 8.413e-01 3.110e-01 9.985e-02 3.162e-02 1.000e-02 3.162e-03 1.000e-03 3.161e-04 5.405e-01 9.502e-01 9.951e-01 9.995e-01 1.000e+00 1.000e+00 1.000e+00 1.000e+00] [ 9.092e-01 5.913e-01 1.986e-01 6.323e-02 2.000e-02 6.325e-03 2.001e-03 6.323e-04 -4.163e-01 8.066e-01 9.800e-01 9.980e-01 1.000e+00 1.000e+00 1.000e+00 1.000e+00] [ 1.411e-01 8.125e-01 2.954e-01 9.473e-02 3.000e-02 9.483e-03 3.000e-03 9.489e-04 -9.902e-01 5.825e-01 9.556e-01 9.956e-01 9.995e-01 1.000e+00 1.000e+00 1.000e+00] [-7.568e-01 9.536e-01 3.894e-01 1.261e-01 3.998e-02 1.265e-02 4.002e-03 1.265e-03 -6.538e-01 3.010e-01 9.209e-01 9.922e-01 9.990e-01 1.000e+00 1.000e+00 1.000e+00]]]

Status
Passed


import numpy as np

def pos_encoding(position: int, d_model: int):
    if position == 0 or d_model <= 0:
        return -1

    pos = np.array(np.arange(position), np.float32)
    ind = np.array(np.arange(d_model), np.float32)
    pos = pos.reshape(position, 1)
    ind = ind.reshape(1, d_model)

    def get_angles(pos, i):
        angles = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angles

    angle1 = get_angles(pos, ind)
    sine = np.sin(angle1[:, 0::2])
    cosine = np.cos(angle1[:, 1::2])
    pos_encoding = np.concatenate([sine, cosine], axis=-1)
    pos_encoding = pos_encoding[np.newaxis, :]
    pos_encoding = np.float16(pos_encoding)
    return pos_encoding
"""