# -*- coding: utf-8 -*-            
# @Author : CharlottE
# @Time : 2024/10/22 14:27

"""2D Translation Matrix Implementation Task: Implement a 2D Translation Matrix Your task is to implement a function
that applies a 2D translation matrix to a set of points. A translation matrix is used to move points in 2D space by a
specified distance in the x and y directions.

Write a function translate_object(points, tx, ty) where points is a list of [x, y] coordinates and tx and ty are the
translation distances in the x and y directions respectively.

The function should return a new list of points after applying the translation matrix.

Example
Example:
import numpy as np

points = [[0, 0], [1, 0], [0.5, 1]]
tx, ty = 2, 3

print(translate_object(points, tx, ty))

# Expected Output:
# [[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]]
"""

import numpy as np

def translate_object(points, tx, ty):
    translated_points = []
    for point in points:
        translated_points.append([point[0] + tx, point[1] + ty])
    translated_points = np.array(translated_points)
    return translated_points

points = [[0, 0], [1, 0], [0.5, 1]]
tx, ty = 2, 3

"""
Test Case 1: Accepted
Input:
import numpy as np

triangle = [[0, 0], [1, 0], [0.5, 1]]
tx, ty = 2, 3
print(translate_object(triangle, tx, ty))
Output:
[[2.  3. ]
 [3.  3. ]
 [2.5 4. ]]
Expected:
[[2.0, 3.0], [3.0, 3.0], [2.5, 4.0]]
Test Case 2: Accepted
Input:
import numpy as np

square = [[0, 0], [1, 0], [1, 1], [0, 1]]
tx, ty = -1, 2
print(translate_object(square, tx, ty))
Output:
[[-1  2]
 [ 0  2]
 [ 0  3]
 [-1  3]]
Expected:
[[-1.0, 2.0], [0.0, 2.0], [0.0, 3.0], [-1.0, 3.0]]
"""

print(translate_object(points, tx, ty))
