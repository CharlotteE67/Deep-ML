# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/11/22 19:45
"""
Implement Gini Impurity Calculation for a Set of Classes
Task: Implement Gini Impurity Calculation
Your task is to implement a function that calculates the Gini Impurity for a set of classes. Gini impurity is commonly used in decision tree algorithms to measure the impurity or disorder within a node.

Write a function gini_impurity(y) that takes in a list of class labels y and returns the Gini Impurity rounded to three decimal places.

Example
Example:
y = [0, 1, 1, 1, 0]
print(gini_impurity(y))

# Expected Output:
# 0.48
"""
from collections import Counter

import numpy as np


def gini_impurity(y):
    """
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
    count = Counter(y)
    ans = [(v/len(y))**2 for k, v in count.items()]
    # print(ans)
    return round(1 - sum(ans), 3)


y = [0, 1, 1, 1, 0]
print(gini_impurity(y))


"""
Test Case 1: Accepted
Input:
y = [0, 0, 0, 0, 1, 1, 1, 1]
print(gini_impurity(y))
Output:
0.5
Expected:
0.5
Test Case 2: Accepted
Input:
y = [0, 0, 0, 0, 0, 1]
print(gini_impurity(y))
Output:
0.278
Expected:
0.278
Test Case 3: Accepted
Input:
y = [0, 1, 2, 2, 2, 1, 2]
print(gini_impurity(y))
Output:
0.571
Expected:
0.571

import numpy as np

def gini_impurity(y: list[int]) -> float:

    classes = set(y)
    n = len(y)

    gini_impurity = 0

    for cls in classes:
        gini_impurity += (y.count(cls)/n)**2

    return round(1-gini_impurity,3)

"""