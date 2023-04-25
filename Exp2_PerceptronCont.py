import numpy as np

import numpy as np
import itertools
def unipolar(net, lambdaa = 0.3):
  return (1/(1+np.exp(-lambdaa*net)))
def bipolar(net, lambdaa = 1):
  return (2/(1+np.exp(-lambdaa*net))) - 1
def predictt(X,weights,type):
  result = np.dot(X,weights)
  if type == "unipolar":
    return unipolar(result)
  else:
    return bipolar(result)
def train(X,y,weights,type,epochs = 100,c=0.5):
  for epoch in range(epochs):
    loss = 0
    for Xi,yi in zip(X,y):
      y_pred = predictt(Xi,weights,type)
      r = yi - y_pred
      loss += abs(r)
      delta_w = c*r*Xi
      weights += delta_w
    print(f'Weights after epoch {epoch} : ',weights)
    if loss == 0:
      break
  print('Learned Weights : ',weights)
  weights = weights.reshape((n+1,1))
  test(X,y,weights,type)
def test(X,y,learned_weights,type):
  results = np.dot(X,learned_weights).flatten()
  print("Actual Values : ",y)
  if type == "unipolar":
    y_pred = np.array([1 if result >=0 else 0 for result in results ])
    print('Predicted Values : ', y_pred)
  if type == "bipolar":
    y_pred = np.array([1 if result >=0 else -1 for result in results ])
    print('Predicted Values : ', y_pred)
n = int(input("enter number of bits  "))
X = np.array([list(i) + [1] for i in itertools.product([0,1],repeat = n)])
weights = input(f'Enter {n} initial weights and 1 bias : ')
weights = np.array([float(weight) for weight in weights.split()],dtype='longdouble')

print()

#AND GATE
print('---- AND GATE USING PERCEPTRON ----')
y = np.array([0] * (2**n))
y[-1] = 1
train(X,y,weights.copy(),'unipolar')

print('---- OR GATE USING PERCEPTRON ----')
y = np.array([1]*(2**n))
y[0] = 0
train(X, y, weights.copy(), 'unipolar')
print()
print()

# 3) NOR GATE UNIPOLAR
print('---- NOR GATE USING PERCEPTRON ----')
y = np.array([0]*(2**n))
y[0] = 1
train(X, y, weights.copy(), 'unipolar')
print()
print()

# 4) NAND GATE UNIPOLAR
print('---- NAND GATE USING PERCEPTRON ----')
y = np.array([1]*(2**n))
y[-1] = 0
train(X, y, weights.copy(), 'unipolar')
print()
print()