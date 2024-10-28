# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/28 12:43
"""
Implement Recall Metric in Binary Classification
Task:
Implement Recall in Binary Classification
Your task is to implement the recall metric in a binary classification setting. Recall is a performance measure that evaluates how
effectively a machine learning model identifies positive instances from all the actual positive cases in a dataset.

You need to write a function recall(y_true, y_pred) that calculates the recall metric. The function should accept two inputs

Your function should return the recall value rounded to three decimal places. If the denominator (TP + FN) is zero,
the recall should be 0.0 to avoid division by zero.

Example
Example:
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

print(recall(y_true, y_pred))

# Expected Output:
# 0.75
"""
import numpy as np


def recall(y_true, y_pred):
    TP, FN = 0, 0
    for i in range(len(y_pred)):
        if y_pred[i] == 1 and y_true[i] == 1:
            TP += 1
        elif y_pred[i] == 0 and y_true[i] == 1:
            FN += 1
    if TP + FN == 0:
        return 0.
    return round(TP / (TP + FN), 3)


y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])

print(recall(y_true, y_pred))

"""
Test Case 1: Accepted
Input:
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([1, 0, 1, 0, 0, 1])
print(recall(y_true, y_pred))
Output:
0.75
Expected:
0.75
Test Case 2: Accepted
Input:
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 0])
y_pred = np.array([1, 0, 0, 0, 0, 1])
print(recall(y_true, y_pred))
Output:
0.333
Expected:
0.333
Test Case 3: Accepted
Input:
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 0])
y_pred = np.array([1, 0, 1, 1, 0, 0])
print(recall(y_true, y_pred))
Output:
1.0
Expected:
1.0
Test Case 4: Accepted
Input:
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([0, 0, 0, 1, 0, 1])
print(recall(y_true, y_pred))
Output:
0.5
Expected:
0.5
Test Case 5: Accepted
Input:
import numpy as np

y_true = np.array([1, 0, 1, 1, 0, 1])
y_pred = np.array([0, 1, 0, 0, 1, 0])
print(recall(y_true, y_pred))
Output:
0.0
Expected:
0.0

import numpy as np

def recall(y_true, y_pred):
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    try:
        return round(tp / (tp + fn), 3)
    except ZeroDivisionError:
        return 0.0


"""