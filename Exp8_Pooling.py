import numpy as np

def pool(matrix, stride, pool_size):
    paddedMatrix = np.array([[0] * paddedMatrix_width for i in range(paddedMatrix_height)])
    height = len(matrix)
    width = len(matrix[0])
    i = 0
    j = 0
    oi = 0
    oj = 0

    while (i + pool_size <= height):
        while (j + pool_size <= width):
            pool_window = []

            for h in range(pool_size):
                for w in range(pool_size):
                    pool_window.append(matrix[i + h][j + w])

            paddedMatrix[oi][oj] = max(pool_window)
            oj += 1
            j = j + stride
        i = i + stride
        oi += 1
        j = 0
        oj = 0
    return paddedMatrix


print("Creating Original Matrix ")

og_height = int(input("Enter the height of matrix"))
og_width = int(input("Enter the width of matrix"))
matrix = [[0] * og_width for i in range(og_height)]

for i in range(og_height):
    for j in range(og_width):
        matrix[i][j] = int(input(f'Enter the values of matrix[{i}][{j}] :'))

pad_amt = int(input("enter the amount of padding"))
pad_val = int(input("Enter the padding value"))
stride = int(input("enter the amount of stride"))
pool_size = int(input("enter the pool size "))

paddedMatrix_width = ((og_width + (2 * pad_amt) - pool_size) // stride) + 1
paddedMatrix_height = ((og_height + (2 * pad_amt) - pool_size) // stride) + 1

matrix = np.array(matrix)
print()

print("Input Matrix : ")
print(matrix)
print()

matrix = np.pad(matrix, pad_amt, constant_values=pad_val)
print("After Padding : ")
print(matrix)
print()

matrix = pool(matrix, stride, pool_size)
print('After Pooling :- ')
print(matrix)