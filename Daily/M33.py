# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/18 13:59

"""
Generate Random Subsets of a Dataset
Write a Python function to generate random subsets of a given dataset. The function should take in a 2D numpy array X,
a 1D numpy array y, an integer n_subsets, and a boolean replacements.
It should return a list of n_subsets random subsets of the dataset, where each subset is a tuple of (X_subset, y_subset).
If replacements is True, the subsets should be created with replacements; otherwise, without replacements.
Example
Example:
    X = np.array([[1, 2],
                  [3, 4],
                  [5, 6],
                  [7, 8],
                  [9, 10]])
    y = np.array([1, 2, 3, 4, 5])
    n_subsets = 3
    replacements = False
    get_random_subsets(X, y, n_subsets, replacements)

    Output:
    [array([[7, 8],
            [1, 2]]),
     array([4, 1])]

    [array([[9, 10],
            [5, 6]]),
     array([5, 3])]

    [array([[3, 4],
            [5, 6]]),
     array([2, 3])]

    Reasoning:
    The function generates three random subsets of the dataset without replacements.
    Each subset includes 50% of the samples (since replacements=False). The samples
    are randomly selected without duplication.
"""
import numpy as np


# def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
#     # Your code here
#     np.random.seed(seed)
#
#     np.random.choice()
#     if not replacements:
#         subsets = [i for i in range(len(X))]
#         np.random.shuffle(subsets)
#         out = []
#         for i in range(n_subsets):
#             i_subset = [[], []]
#             for j in range(i*len(subsets)//n_subsets, (i+1)*len(subsets)//n_subsets):
#                 i_subset[0].append(X[subsets[j]])
#                 i_subset[1].append(y[subsets[j]])
#                 # i_subset.append([X[subsets[j]], y[subsets[j]]])
#             out.append(i_subset)
#     else:
#         out = []
#         for _ in range(n_subsets):
#             subset = [[], []]
#             for i in range(len(X)//n_subsets):
#                 i_rand = np.random.randint(low=0, high=len(X)-1)
#                 subset[0].append(X[i_rand])
#                 subset[1].append(y[i_rand])
#                 # subset.append([X[i_rand], y[i_rand]])
#             out.append(subset)
#     return out

import numpy as np


def get_random_subsets(X, y, n_subsets, replacements=True, seed=42):
    np.random.seed(seed)

    n, m = X.shape

    subset_size = n if replacements else n // 2
    idx = np.array([np.random.choice(n, subset_size, replace=replacements) for _ in range(n_subsets)])
    print(idx)
    # convert all ndarrays to lists
    return [(X[idx][i].tolist(), y[idx][i].tolist()) for i in range(n_subsets)]

# X = np.array([[1, 2],
#                   [3, 4],
#                   [5, 6],
#                   [7, 8],
#                   [9, 10]])
# y = np.array([1, 2, 3, 4, 5])

X = np.array([[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
y = np.array([6, 5, 4, 3, 2, 1])

n_subsets = 2
replacements = False
ans = get_random_subsets(X, y, n_subsets, replacements)
print(ans)


"""
Test Case 1: Accepted
Input:
 
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]])
y = np.array([1, 2, 3, 4, 5])
print(get_random_subsets(X,y, 3, False, seed=42))
Output:
[([[3, 4], [9, 10]], [2, 5]), ([[7, 8], [3, 4]], [4, 2]), ([[3, 4], [1, 2]], [2, 1])]
Expected:
[[[3, 4], [9, 10]], [2, 5], [[7, 8], [3, 4]], [4, 2], [[3, 4], [1, 2]], [2, 1]]
Test Case 2: Accepted
Input:
X = np.array([[1, 1], [2, 2], [3, 3], [4, 4]])
y = np.array([10, 20, 30, 40])
print(get_random_subsets(X, y, 1, True, seed=42))
Output:
[([[3, 3], [4, 4], [1, 1], [3, 3]], [30, 40, 10, 30])]
Expected:
[([[3, 3], [4, 4], [1, 1], [3, 3]], [30, 40, 10, 30])]
"""
