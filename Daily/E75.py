# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/12/16 12:45
"""
Generate a Confusion Matrix for Binary Classification
Task: Generate a Confusion Matrix
Your task is to implement the function confusion_matrix(data) that generates a confusion matrix for a binary classification problem.
The confusion matrix provides a summary of the prediction results on a classification problem,
allowing you to visualize how many data points were correctly or incorrectly labeled.

Input: A list of lists, where each inner list represents a pair [y_true, y_pred] for one observation.

Output: A 2*2 confusion matrix.

Example
Example:
data = [[1, 1], [1, 0], [0, 1], [0, 0], [0, 1]]
print(confusion_matrix(data))
Output:
[[1, 1], [2, 1]]
"""
from collections import Counter

def confusion_matrix(data):
    # Implement the function here
    TP, FN, FP, TN = 0, 0, 0, 0
    for pair in data:
        if pair == [1, 1]:
            TP += 1
        elif pair == [1, 0]:
            FN += 1
        elif pair == [0, 1]:
            FP += 1
        elif pair == [0, 0]:
            TN += 1
    return [[TP, FN], [FP, TN]]

data = [[1, 1], [1, 0], [0, 1], [0, 0], [0, 1]]
print(confusion_matrix(data))

"""
Test Case 1: Accepted
Input:
data = [[1, 1], [1, 0], [0, 1], [0, 0], [0, 1]]
print(confusion_matrix(data))
Output:
[[1, 1], [2, 1]]
Expected:
[[1, 1], [2, 1]]
Test Case 2: Accepted
Input:
data = [[0, 1], [1, 0], [1, 1], [0, 1], [0, 0], [1, 0], [0, 1], [1, 1], [0, 0], [1, 0], [1, 1], [0, 0], [1, 0], [0, 1], [1, 1], [1, 1], [1, 0]]
print(confusion_matrix(data))
Output:
[[5, 5], [4, 3]]
Expected:
[[5, 5], [4, 3]]
Test Case 3: Accepted
Input:
data = [[0, 1], [0, 1], [0, 0], [0, 1], [0, 0], [0, 1], [0, 1], [0, 0], [1, 0], [0, 1], [1, 0], [0, 0], [0, 1], [0, 1], [0, 1], [1, 0]]
print(confusion_matrix(data))
Output:
[[0, 3], [9, 4]]
Expected:
[[0, 3], [9, 4]]


from collections import Counter

def confusion_matrix(data):
    # Count all occurrences
    counts = Counter(tuple(pair) for pair in data)
    # Get metrics
    TP, FN, FP, TN = counts[(1, 1)], counts[(1, 0)], counts[(0, 1)], counts[(0, 0)]
    # Define matrix and return
    confusion_matrix = [[TP, FN], [FP, TN]]
    return confusion_matrix

"""