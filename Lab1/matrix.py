# mat = [[1, 2], [3, 4]]
# vec = [7, 8]

# dot = [0 for elem in range(len(mat))]

# for i in range(len(mat)):
#     for j in range(len(mat[0])):
#         dot[i] += mat[i][j] * vec[i]

# print(dot)

import numpy as np

mat = np.matrix([[1,2,3,4], [11,12,13,14], [21,22,23,24]])
vec = np.array([2, -5, 7, 10])

# print(mat[:2, -2:])
# print(vec[-2:])

vec1 = np.random.rand(3)
vec2 = np.random.rand(3)

# print(vec1)
# print(vec2)

# if np.sum(vec1) > np.sum(vec2):
#     print("vec1")
# else:
#     print("vec2")

# print(np.sqrt(vec1))

matr = np.random.rand(5,3)

# print(np.transpose(matr))
print(np.linalg.det(matr))
print(np.linalg.inv(matr))