import common

def matrix_from_array(arr, tup):
    rows = tup[0]
    cols = tup[1]

    if len(arr) != (rows * cols):
        raise Exception("Invalid dimensions")

    mat = []

    for j in range(rows):
        row = arr[j * cols: (j + 1) * cols]
        mat.append(row)

    return mat

def calculate_determinant(mat):
    det = 0

    if len(mat) == 2:
        return mat[0][0] * mat[1][1] - mat[0][1] * mat[1][0]

    for i in range(3):
        det += (mat[i][0] * mat[(i+1)%3][1]* mat[(i+2)%3][2])
        det -= (mat[i][2] * mat[(i+1)%3][1]* mat[(i+2)%3][0])

    return det

def get_matrix_of_minors(mat):
    matr = []
    
    for i in range(3):
        matr_row = []
        for j in range(3):
            det_matr = []
            for k in range(3):
                if k != i:
                    pos_matr = []
                    pos_matr.extend(mat[k][0:j])
                    pos_matr.extend(mat[k][j+1:])
                    det_matr.append(pos_matr)
            det = calculate_determinant(det_matr)
            if (i + j)%2:
                det = -det

            matr_row.append(det)

        matr.append(matr_row)

    return matr

def get_inverse_of_matrix(mat):
    det = calculate_determinant(mat)
    matr = get_matrix_of_minors(mat)

    for i in range(3):
        for j in range(i + 1, 3):
            matr[i][j], matr[j][i] = matr[j][i], matr[i][j]

    for i in range(3):
        for j in range(3):
            matr[i][j] = 1.0/det * matr[i][j]

    return matr;

def matrix_multiply(mat1, mat2):
    matr_result = [[0],[0],[0]]
    
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat1)):
                matr_result[i][j] += mat1[i][k] * mat2[k][j]

    return matr_result

(constants, r) = common.get_values_array()

matrix_a = matrix_from_array(constants, (3,3))
matrix_b = matrix_from_array(r, (3,1))

x = matrix_multiply(get_inverse_of_matrix(matrix_a), matrix_b)

common.print_values(x)