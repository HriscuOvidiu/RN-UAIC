import numpy as np
import common

(constants, r) = common.get_values_array()

matrix_a = np.reshape(constants, (3, 3))
matrix_b = np.reshape(r, (3,1))

x = np.dot(np.linalg.inv(matrix_a), matrix_b)

common.print_values(x)