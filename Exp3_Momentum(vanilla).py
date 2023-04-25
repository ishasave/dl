import numpy as np


def predict(X, weights, l=1):
    result = np.dot(X, weights)
    return (1 / (1 + np.exp(-l * result)))


def calculate_derivate(X, y, y_pred):
    return ((y_pred - y) * (y_pred) * (1 - y_pred) * X)


def calculate_error(y, y_pred):
    return np.square(y_pred - y)


def momentum_vanilla_gradient_descent(X, y, weights, epochs=1000, learning_rate=0.01, gamma=0.9):
    velocity = 0
    for epoch in range(epochs):
        dw = 0
        error = 0
        for Xi, yi in zip(X, y):
            y_pred = predict(Xi, weights)
            dw += calculate_derivate(Xi, yi, y_pred)
            error += calculate_error(yi, y_pred)

        velocity = (gamma * velocity) + (learning_rate * dw)

        weights -= velocity

        error /= (2 * len(X))

        if (epoch + 1) % 50 == 0:
            print(f'Weights after epoch {epoch + 1} :', weights)
            print(f'Error after epoch {epoch + 1} :', error)


def momentum_stochastic_gradient_descent(X, y, weights, epochs=1000, learning_rate=0.01, gamma=0.9):
    velocity = 0
    for epoch in range(epochs):

        error = 0
        for Xi, yi in zip(X, y):
            y_pred = predict(Xi, weights)
            dw = calculate_derivate(Xi, yi, y_pred)
            error += calculate_error(yi, y_pred)

            velocity = (gamma * velocity) + (learning_rate * dw)

            weights -= velocity

        error /= (2 * len(X))

        if (epoch + 1) % 50 == 0:
            print(f'Weights after epoch {epoch + 1} :', weights)
            print(f'Error after epoch {epoch + 1} :', error)


def momentum_minibatch_gradient_descent(X, y, weights, epochs=1000, learning_rate=0.01, gamma=0.9, batch=2):
    velocity = 0
    for epoch in range(epochs):
        dw = 0
        error = 0
        i = 0
        for Xi, yi in zip(X, y):
            y_pred = predict(Xi, weights)
            dw += calculate_derivate(Xi, yi, y_pred)
            i += 1
            error += calculate_error(yi, y_pred)
            if i % batch == 0 or i == len(X):
                velocity = (gamma * velocity) + (learning_rate * dw)
                weights -= velocity
                # important
                dw = 0

        error /= (2 * len(X))

        if (epoch + 1) % 50 == 0:
            print(f'Weights after epoch {epoch + 1} :', weights)
            print(f'Error after epoch {epoch + 1} :', error)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

weights = np.array([-1, 1], dtype='longdouble')

momentum_vanilla_gradient_descent(X, y, weights)