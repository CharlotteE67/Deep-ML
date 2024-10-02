# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/2 14:31
"""
Simple Convolutional 2D Layer
In this problem, you need to implement a 2D convolutional layer in Python. This function will process an input matrix
using a specified convolutional kernel, padding, and stride.
Example
Example:
import numpy as np

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])

padding = 1
stride = 2

output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)
# Expected Output:
# [[  3.   9.]
#  [ 11.  17.]]
"""
import numpy as np


def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
    input_height, input_width = input_matrix.shape
    kernel_height, kernel_width = kernel.shape

    padded_matrix = np.pad(input_matrix, pad_width=(padding, padding), mode='constant')
    # print(padded_matrix)
    padded_height, padded_width = padded_matrix.shape
    output_height, output_width = (padded_height - kernel_height) // stride + 1, (
                padded_width - kernel_width) // stride + 1

    output_matrix = np.zeros((output_height, output_width))
    # print(output_matrix)

    for i in range(0, padded_height - kernel_height + 1, stride):
        for j in range(0, padded_width - kernel_width + 1, stride):
            # print(padded_matrix[i:i + kernel_height, j:j + kernel_width], kernel)
            # print(padded_matrix[i:i + kernel_height, j:j + kernel_width] * kernel)
            output_matrix[i // stride, j // stride] = np.sum(padded_matrix[i:i + kernel_height, j:j + kernel_width] * kernel)

    # Your code here

    return output_matrix

input_matrix = np.array([
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12],
    [13, 14, 15, 16]
])

kernel = np.array([
    [1, 0],
    [-1, 1]
])
#
# kernel_height, kernel_width = kernel.shape
padding = 1
stride = 2
output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)
# padded_input = np.pad(input_matrix, ((padding, padding), (padding, padding)), mode='constant')
# print(padded_input)
# input_height_padded, input_width_padded = padded_input.shape
#
# output_height = (input_height_padded - kernel_height) // stride + 1
# output_width = (input_width_padded - kernel_width) // stride + 1
#
# output_matrix = np.zeros((output_height, output_width))
#
# for i in range(output_height):
#     for j in range(output_width):
#         region = padded_input[i*stride:i*stride + kernel_height, j*stride:j*stride + kernel_width]
#         output_matrix[i, j] = np.sum(region * kernel)
#
# print(output_matrix)

input_matrix = np.array([
    [1., 2., 3., 4., 5.],
    [6., 7., 8., 9., 10.],
    [11., 12., 13., 14., 15.],
    [16., 17., 18., 19., 20.],
    [21., 22., 23., 24., 25.],
])
kernel = np.array([
    [1., 2.],
    [3., -1.],
])
padding, stride = 0, 1
expected = np.array([
    [ 16., 21., 26., 31.],
    [ 41., 46., 51., 56.],
    [ 66., 71., 76., 81.],
    [ 91., 96., 101., 106.],
])
output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)

"""
Test Case 1: Accepted
Input:
input_matrix = np.array([
    [1., 2., 3., 4., 5.],
    [6., 7., 8., 9., 10.],
    [11., 12., 13., 14., 15.],
    [16., 17., 18., 19., 20.],
    [21., 22., 23., 24., 25.],
])
kernel = np.array([
    [1., 2.],
    [3., -1.],
])
padding, stride = 0, 1
expected = np.array([
    [ 16., 21., 26., 31.],
    [ 41., 46., 51., 56.],
    [ 66., 71., 76., 81.],
    [ 91., 96., 101., 106.],
])
output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)
Output:
[[ 16.  21.  26.  31.]
 [ 41.  46.  51.  56.]
 [ 66.  71.  76.  81.]
 [ 91.  96. 101. 106.]]
Expected:
[[ 16.,  21.,  26.,  31.],
 [ 41.,  46.,  51.,  56.],
 [ 66.,  71.,  76.,  81.],
 [ 91.,  96., 101., 106.]]
Test Case 2: Accepted
Input:
input_matrix = np.array([
    [1., 2., 3., 4., 5.],
    [6., 7., 8., 9., 10.],
    [11., 12., 13., 14., 15.],
    [16., 17., 18., 19., 20.],
    [21., 22., 23., 24., 25.],
])
kernel = np.array([
    [.5, 3.2],
    [1., -1.],
])
padding, stride = 2, 2
expected = np.array([
        [ -1., 1., 3., 5., 7., 15.],
        [ -4., 16., 21., 26., 31., 35.],
        [  1., 41., 46., 51., 56., 55.],
        [  6., 66., 71., 76., 81., 75.],
        [ 11., 91., 96., 101., 106., 95.],
        [ 42., 65., 68., 71., 74.,  25.],
    ])
output = simple_conv2d(input_matrix, kernel, padding, stride)
print(output)
Output:
[[ 0.   0.   0.   0. ]
 [ 0.   5.9 13.3 12.5]
 [ 0.  42.9 50.3 27.5]
 [ 0.  80.9 88.3 12.5]]
Expected:
[[ 0.,   0.,   0.,   0. ],
 [ 0.,   5.9, 13.3, 12.5],
 [ 0.,  42.9, 50.3, 27.5],
 [ 0.,  80.9, 88.3, 12.5],]
"""