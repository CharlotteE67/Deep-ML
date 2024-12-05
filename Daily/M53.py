# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/12/5 14:54
"""
Implement Self-Attention Mechanism
Task: Implement the Self-Attention Mechanism
Your task is to implement the self-attention mechanism, which is a fundamental component of transformer models,
widely used in natural language processing and computer vision tasks.
The self-attention mechanism allows a model to dynamically focus on different parts of the input sequence when generating a contextualized representation.

Your function should return the self-attention output as a numpy array.

Example
Example:
import numpy as np

X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)

print(output)

# Expected Output:
# [[1.660477 2.660477]
#  [2.339523 3.339523]]
"""

import numpy as np


def self_attention(Q, K, V):
    # print(type(Q), type(K), type(V), Q.shape[0])
    attention_output = np.matmul(Q, K.T) / np.sqrt(Q.shape[-1])
    attention_weights = np.exp(attention_output) / np.sum(np.exp(attention_output), axis=1, keepdims=True)
    attention_output = np.matmul(attention_weights, V)
    return attention_output

def compute_qkv(x, wq, wk, wv):
    # return np.dot(x, wq), np.dot(x, wk), np.dot(x, wv)
    return x@wq, x@wk, x@wv

# X = np.array([[1, 0], [0, 1]])
# W_q = np.array([[1, 0], [0, 1]])
# W_k = np.array([[1, 0], [0, 1]])
# W_v = np.array([[1, 2], [3, 4]])
X = np.array([[1, 1], [1, 0]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
print(Q, K, V)
output = self_attention(Q, K, V)

print(output)

"""
import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    d_k = Q.shape[1]
    scores = np.matmul(Q, K.T) / np.sqrt(d_k)
    attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=1, keepdims=True)
    attention_output = np.matmul(attention_weights, V)
    return attention_output



Test Case 1: Accepted
Input:
import numpy as np

X = np.array([[1, 0], [0, 1]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)
print(output)
Output:
[[1.6604769 2.6604769]
 [2.3395231 3.3395231]]
Expected:
[[1.660477, 2.660477], [2.339523, 3.339523]]
Test Case 2: Accepted
Input:
import numpy as np

X = np.array([[1, 1], [1, 0]])
W_q = np.array([[1, 0], [0, 1]])
W_k = np.array([[1, 0], [0, 1]])
W_v = np.array([[1, 2], [3, 4]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)
print(output)
Output:
[[3.00928465 4.6790462 ]
 [2.5        4.        ]]
Expected:
[[3.00928465, 4.6790462], [2.5, 4.0]]
Test Case 3: Accepted
Input:
import numpy as np

X = np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0]])
W_q = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
W_k = np.array([[1, 1, 0], [0, 1, 1], [1, 0, 1]])
W_v = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

Q, K, V = compute_qkv(X, W_q, W_k, W_v)
output = self_attention(Q, K, V)
print(output)
Output:
[[ 8.         10.         12.        ]
 [ 8.61987385 10.61987385 12.61987385]
 [ 7.38012615  9.38012615 11.38012615]]
Expected:
[[8.0, 10.0, 12.0], [8.61987385, 10.61987385, 12.61987385], [7.38012615, 9.38012615, 11.38012615]]
"""