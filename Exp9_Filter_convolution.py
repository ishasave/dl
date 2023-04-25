import numpy as np

def convolve(matrix, filter, stride):
    op_matrix = np.array([[0] * filteredMatrix_width for i in range(filteredMatrix_height)])

    height = len(matrix)
    width = len(matrix[0])

    i = 0
    j = 0
    oi = 0
    oj = 0

    while (i + filter_height <= height):
        while (j + filter_width <= width):
            window = []
            for h in range(filter_height):
                for w in range(filter_width):
                    window.append(matrix[i + h][j + w])

            op_matrix[oi][oj] = np.dot(np.array(window), filter.ravel())
            oj += 1
            j = j + stride
        i = i + stride
        oi += 1
        j = 0
        oj = 0
    return op_matrix


og_height = int(input("Enter the height of matrix"))
og_width = int(input("Enter the width of matrix"))
matrix = [[0] * og_width for i in range(og_height)]

for i in range(og_height):
    for j in range(og_width):
        matrix[i][j] = int(input(f'Enter the values of matrix[{i}][{j}] :'))

filter_height = int(input("Filter Height"))
filter_width = int(input("Filter Width"))

filter = [[0] * filter_width for i in range(filter_height)]

for i in range(filter_height):
    for j in range(filter_width):
        filter[i][j] = int(input(f'Enter value of filter[{i}][{j}] : '))

pad_amt = int(input("enter the amount of padding"))
pad_val = int(input("Enter the padding value"))
stride = int(input("enter the amount of stride"))

filteredMatrix_width = ((og_width + (2 * pad_amt) - filter_width) // stride) + 1
filteredMatrix_height = ((og_height + (2 * pad_amt) - filter_height) // stride) + 1

filter = np.array(filter)
matrix = np.array(matrix)

print("Input matrix")
print(matrix)
print()
print("Filter : ")
print(filter)
print()
matrix = np.pad(matrix, pad_amt, constant_values=pad_val)
print("After Padding : ")
print(matrix)
print()
filteredOp = convolve(matrix, filter, stride)
print("After Filter ")
print(filteredOp)