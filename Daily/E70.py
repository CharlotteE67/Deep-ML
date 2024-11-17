# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/11/17 15:51
"""
Calculate Image Brightness
Task: Image Brightness Calculator
In this task, you will implement a function calculate_brightness(img) that calculates the average brightness of a grayscale image.
The image is represented as a 2D matrix, where each element represents a pixel value between 0 (black) and 255 (white).

Your Task: Implement the function calculate_brightness(img) to return the average brightness rounded to two decimal places.
If the image matrix is empty, has inconsistent row lengths, or contains invalid pixel values (outside the range 0-255), the function should return -1.

Example
Example:
img = [
    [100, 200],
    [50, 150]
]
print(calculate_brightness(img))
Output: 125.0
"""


def calculate_brightness(img):
    # Write your code here
    if not img or any([any(item< 0 or item > 255 for item in img[i]) for i in range(len(img))]) or len(set([len(img[i]) for i in range(len(img))])) != 1:
        return -1

    m, n = len(img), len(img[0])
    return round(sum([sum(item) for item in img]) / (m * n), 2)

img = [
    [100, 200],
    [50, 150]
]
print(calculate_brightness(img))


"""
Test Case 1: Accepted
Input:
# Test empty image
print(calculate_brightness([]))
Output:
-1
Expected:
-1
Test Case 2: Accepted
Input:
# Test invalid dimensions
print(calculate_brightness([[100, 200], [150]]))
Output:
-1
Expected:
-1
Test Case 3: Accepted
Input:
# Test invalid pixel values
print(calculate_brightness([[100, 300]]))
Output:
-1
Expected:
-1
Test Case 4: Accepted
Input:
# Test valid cases
print(calculate_brightness([[128]]))
Output:
128.0
Expected:
128.0
Test Case 5: Accepted
Input:
# Another valid case
print(calculate_brightness([[100, 200], [50, 150]]))
Output:
125.0
Expected:
125.0

def calculate_brightness(img):
    # Check if image is empty or has no columns
    if not img or not img[0]:
        return -1

    rows, cols = len(img), len(img[0])

    # Check if all rows have same length and values are valid
    for row in img:
        if len(row) != cols:
            return -1
        for pixel in row:
            if not 0 <= pixel <= 255:
                return -1

    # Calculate average brightness
    total = sum(sum(row) for row in img)
    return round(total / (rows * cols), 2)

"""