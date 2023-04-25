import numpy as np
from numpy.matrixlib.defmatrix import matrix

def makeMatrix(arr,n):
  l = []
  for i in range(0,n):
    l.append(arr[i])
  matrix.append(l)

def generateAllBinaryStrings(n,arr,i):
  if i == n:
    makeMatrix(arr,n)
    return
  arr[i] = 0
  generateAllBinaryStrings(n,arr,i+1)

  arr[i] = 1
  generateAllBinaryStrings(n,arr,i+1)


# AND GATE
col = int(input("Enter the number of inputs"))
row = pow(2, col)

arr = [None] * col
matrix = [[]]
generateAllBinaryStrings(col, arr, 0)

print("Output of AND Gate : ")

W = np.ones((col,), dtype='int')

for i in range(1, row + 1):
    sum = 0
    for j in range(col):
        sum += matrix[i][j] * W[j]
    if sum >= col:
        for r in range(col):
            print(matrix[i][r], end=' ')
        print(1)
    else:
        for r in range(col):
            print(matrix[i][r], end=' ')
        print(0)


#NAND GATE

arr = [None] * col
matrix = [[]]
generateAllBinaryStrings(col,arr,0)

print("Output of Nand Gate : ")

W = -1*(np.ones((col,) , dtype='int'))

for i in range(1,row+1):
  sum = 0
  for j in range(col):
    sum += matrix[i][j] * W[j]
  if sum >= -(col-1):
    for r in range(col):
      print(matrix[i][r],end =' ')
    print(1)
  else:
    for r in range(col):
      print(matrix[i][r],end=' ')
    print(0)

#NOR GATE

arr = [None] * col
matrix = [[]]
generateAllBinaryStrings(col,arr,0)

print("Output of NOR Gate : ")

W = -1*(np.ones((col,) , dtype='int'))

for i in range(1,row+1):
  sum = 0
  for j in range(col):
    sum += matrix[i][j] * W[j]
  if sum >= 0:
    for r in range(col):
      print(matrix[i][r],end =' ')
    print(1)
  else:
    for r in range(col):
      print(matrix[i][r],end=' ')
    print(0)


#NOT

x = [0,1]
w = -1
t = 0

print("Output of OR Gate : ")
for i in range(len(x)):
  if x[i]*w >= t:
    print("~",x[i],":",1)
  else:
    print("~",x[i],":",0)