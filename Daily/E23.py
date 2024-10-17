# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/17 14:50
"""Softmax Activation Function Implementation (easy) Write a Python function that computes the softmax activation for
a given list of scores. The function should return the softmax values as a list, each rounded to four decimal places.
Example
Example:
    input: scores = [1, 2, 3]
    output: [0.0900, 0.2447, 0.6652]
    reasoning: The softmax function converts a list of values into a probability distribution.
    The probabilities are proportional to the exponential of each element divided by the sum of the exponentials of all elements in the list.

"""
import math

from typing import List


def softmax(scores: List[float]) -> List[float]:
    # Your code here
    probabilities = [math.exp(s) for s in scores]
    probabilities = [item/sum(probabilities) for item in probabilities]
    return probabilities

print(softmax(scores = [1, 2, 3]))

"""
Test Case 1: Accepted
Input:
print(softmax([1, 2, 3]))
Output:
[0.09003057317038046, 0.24472847105479767, 0.6652409557748219]
Expected:
[0.09, 0.2447, 0.6652]
Test Case 2: Accepted
Input:
print(softmax([1, 1, 1]))
Output:
[0.3333333333333333, 0.3333333333333333, 0.3333333333333333]
Expected:
[0.3333, 0.3333, 0.3333]
Test Case 3: Accepted
Input:
print(softmax([-1, 0, 5]))
Output:
[0.0024561149044509535, 0.006676412513376451, 0.9908674725821726]
Expected:
[0.0025, 0.0067, 0.9909]

import math
def softmax(scores: list[float]) -> list[float]:
    exp_scores = [math.exp(score) for score in scores]
    sum_exp_scores = sum(exp_scores)
    probabilities = [round(score / sum_exp_scores, 4) for score in exp_scores]
    return probabilities
"""