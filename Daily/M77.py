# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2025/1/11 15:04
"""
Calculate Performance Metrics for a Classification Model
Task: Implement Performance Metrics Calculation
In this task, you are required to implement a function performance_metrics(actual, predicted) that computes various p
erformance metrics for a binary classification problem. These metrics include:

Confusion Matrix
Accuracy
F1 Score
Specificity
Negative Predictive Value
The function should take in two lists:

actual: The actual class labels (1 for positive, 0 for negative).
predicted: The predicted class labels from the model.
Output
The function should return a tuple containing:

confusion_matrix: A 2x2 matrix.
accuracy: A float representing the accuracy of the model.
f1_score: A float representing the F1 score of the model.
specificity: A float representing the specificity of the model.
negative_predictive_value: A float representing the negative predictive value.
Constraints
All elements in the actual and predicted lists must be either 0 or 1.
Both lists must have the same length.
Example:
Input:
actual = [1, 0, 1, 0, 1]
predicted = [1, 0, 0, 1, 1]
print(performance_metrics(actual, predicted))
Output:
([[2, 1], [1, 1]], 0.6, 0.667, 0.5, 0.5)
Reasoning:
The function calculates the confusion matrix, accuracy, F1 score, specificity, and negative predictive value based on the input labels.
The resulting values are rounded to three decimal places as required.

"""
from typing import List


def performance_metrics(actual: List[int], predicted: List[int]) -> tuple:
    # Implement your code here
    TP, FN, FP, TN = 0, 0, 0, 0

    for i in range(len(actual)):
        if actual[i] == 1:
            if actual[i] == predicted[i]:
                TP += 1
            else:
                FN += 1
        else:
            if actual[i] == predicted[i]:
                TN += 1
            else:
                FP += 1
    confusion_matrix = [[TP, FN], [FP, TN]]
    accuracy = (TP + TN) / (TP + FN + TN + FP)
    precision = TP / (TP + FP)
    negativePredictive = TN / (TN + FN)
    recall = TP / (TP + FN)
    specificity = TN / (TN + FP)
    f1 = 2 * precision * recall / (precision + recall)

    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)


actual = [1, 0, 1, 0, 1]
predicted = [1, 0, 0, 1, 1]
print(performance_metrics(actual, predicted))

"""
Test Case
actual = [1, 0, 1, 0, 1] predicted = [1, 0, 0, 1, 1] print(performance_metrics(actual, predicted))

Expected Output
([[2, 1], [1, 1]], 0.6, 0.667, 0.5, 0.5)

Actual Output
([[2, 1], [1, 1]], 0.6, 0.667, 0.5, 0.5)

Status
Passed

Test Case
actual = [1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 1, 1] predicted = [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0] print(performance_metrics(actual, predicted))

Expected Output
([[6, 4], [2, 7]], 0.684, 0.667, 0.778, 0.636)

Actual Output
([[6, 4], [2, 7]], 0.684, 0.667, 0.778, 0.636)

Status
Passed


from collections import Counter

def performance_metrics(actual: list[int], predicted: list[int]) -> tuple:
    data = list(zip(actual, predicted))
    counts = Counter(tuple(pair) for pair in data)
    TP, FN, FP, TN = counts[(1, 1)], counts[(1, 0)], counts[(0, 1)], counts[(0, 0)]
    confusion_matrix = [[TP, FN], [FP, TN]]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)
    negativePredictive = TN / (TN + FN)
    specificity = TN / (TN + FP)
    return confusion_matrix, round(accuracy, 3), round(f1, 3), round(specificity, 3), round(negativePredictive, 3)

"""