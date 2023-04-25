import numpy as np
def predict(Xi,weights,l=1):
  result = np.dot(Xi,weights)
  return (1/(1+np.exp(-l * result)))
def calculate_derivate(Xi,yi,y_pred):
  return((y_pred-yi) * y_pred * (1-y_pred) * Xi)
def calculate_error(yi,y_pred):
  return np.square(y_pred - yi)

def rms_vanilla_gd(X, y, weights, epochs=500, learning_rate=0.01, epsilon=1e-8, beta=0.5, batch=2):
    velocity = 0
    for epoch in range(epochs):
        error = 0
        dw = 0
        for Xi, yi in zip(X, y):
            y_pred = predict(Xi, weights)
            dw += calculate_derivate(Xi, yi, y_pred)
            error += calculate_error(yi, y_pred)

        # note this change in velocity update
        velocity = (beta * velocity) + ((1 - beta) * (dw ** 2))
        # note this weight update similar to adagrad
        weights -= ((learning_rate * dw) / (np.sqrt(velocity + epsilon)))
        error /= (2 * len(X))
        if (epoch + 1) % 50 == 0:
            print(f'Weights after epoch {epoch + 1} : ', weights)
            print(f'Error after epoch {epoch + 1} : ', error)


def rms_stochastic_gd(X, y, weights, epochs=500, learning_rate=0.01, epsilon=1e-8, beta=0.5, batch=2):
    velocity = 0
    for epoch in range(epochs):
        error = 0

        for Xi, yi in zip(X, y):
            y_pred = predict(Xi, weights)
            dw = calculate_derivate(Xi, yi, y_pred)
            error += calculate_error(yi, y_pred)

            # note this change in velocity update
            velocity = (beta * velocity) + ((1 - beta) * (dw ** 2))
            # note this weight update similar to adagrad
            weights -= ((learning_rate * dw) / (np.sqrt(velocity + epsilon)))
        error /= (2 * len(X))
        if (epoch + 1) % 50 == 0:
            print(f'Weights after epoch {epoch + 1} : ', weights)
            print(f'Error after epoch {epoch + 1} : ', error)


def rms_minibatch_gd(X, y, weights, epochs=500, learning_rate=0.01, epsilon=1e-8, beta=0.5, batch=2):
    velocity = 0
    for epoch in range(epochs):
        error = 0
        dw = 0
        i = 0
        for Xi, yi in zip(X, y):
            y_pred = predict(Xi, weights)
            dw += calculate_derivate(Xi, yi, y_pred)
            i += 1
            error += calculate_error(yi, y_pred)
            if i % batch == 0 or i == len(X):
                # note this change in velocity update
                velocity = (beta * velocity) + ((1 - beta) * (dw ** 2))
                # note this weight update similar to adagrad
                weights -= ((learning_rate * dw) / (np.sqrt(velocity + epsilon)))
                dw = 0

        error /= (2 * len(X))
        if (epoch + 1) % 50 == 0:
            print(f'Weights after epoch {epoch + 1} : ', weights)
            print(f'Error after epoch {epoch + 1} : ', error)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

weights = np.array([-1, 1], dtype='longdouble')

rms_vanilla_gd(X, y, weights)