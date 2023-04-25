import numpy as np

def predict(Xi, weights, l=1):
    result = np.dot(Xi, weights)
    return (1 / (1 + np.exp(-l * result)))


def calculate_derivate(Xi, yi, y_pred):
    return ((y_pred - yi) * y_pred * (1 - y_pred) * Xi)


def calculate_error(yi, y_pred):
    return np.square(y_pred - yi)


def adam_vanilla_gd(X, y, weights, epochs=500, learning=0.01, epsilon=1e-8, beta1=0.9, beta2=0.999, batch=2):
    # new
    momentum = 0
    velocity = 0
    for epoch in range(epochs):
        dw = 0
        error = 0
        for Xi, yi in zip(X, y):
            y_pred = predict(Xi, weights)
            dw += calculate_derivate(Xi, yi, y_pred)
            error += calculate_error(yi, y_pred)

        # note the additions - momentum and  velocity
        momentum = (beta1 * momentum) + ((1 - beta1) * dw)
        velocity = (beta2 * velocity) + ((1 - beta2) * (dw ** 2))
        step = epoch + 1
        momentum_cap = momentum / (1 - (beta1 ** step))
        velocity_cap = velocity / (1 - (beta2 ** step))
        weights -= ((learning * momentum_cap) / (np.sqrt(velocity_cap + epsilon)))

        error /= (2 * len(X))

        if (epoch + 1) % 50 == 0:
            print(f'Weights after epoch {epoch + 1}', weights)
            print(f'Error after epoch {epoch + 1}', error)


def adam_stochastic_gd(X, y, weights, epochs=500, learning=0.01, epsilon=1e-8, beta1=0.9, beta2=0.999, batch=2):
    # new
    momentum = 0
    velocity = 0
    # note in stochastic
    step = 0
    for epoch in range(epochs):

        error = 0
        for Xi, yi in zip(X, y):
            y_pred = predict(Xi, weights)
            dw = calculate_derivate(Xi, yi, y_pred)
            error += calculate_error(yi, y_pred)

            # note the additions - momentum and  velocity
            momentum = (beta1 * momentum) + ((1 - beta1) * dw)
            velocity = (beta2 * velocity) + ((1 - beta2) * (dw ** 2))

            # change of step updation in stiochastic
            step += 1

            momentum_cap = momentum / (1 - (beta1 ** step))
            velocity_cap = velocity / (1 - (beta2 ** step))
            weights -= ((learning * momentum_cap) / (np.sqrt(velocity_cap + epsilon)))

        error /= (2 * len(X))

        if (epoch + 1) % 50 == 0:
            print(f'Weights after epoch {epoch + 1}', weights)
            print(f'Error after epoch {epoch + 1}', error)


def adam_minibatch_gd(X, y, weights, epochs=500, learning=0.01, epsilon=1e-8, beta1=0.9, beta2=0.999, batch=2):
    # new
    momentum = 0
    velocity = 0

    # see step initialisation
    step = 0
    for epoch in range(epochs):
        dw = 0
        i = 0
        error = 0
        for Xi, yi in zip(X, y):
            y_pred = predict(Xi, weights)
            dw += calculate_derivate(Xi, yi, y_pred)
            i += 1
            error += calculate_error(yi, y_pred)
            if i % batch == 0 or i == len(X):
                # note the additions - momentum and  velocity
                momentum = (beta1 * momentum) + ((1 - beta1) * dw)
                velocity = (beta2 * velocity) + ((1 - beta2) * (dw ** 2))

                # note change in step updation
                step += 1
                momentum_cap = momentum / (1 - (beta1 ** step))
                velocity_cap = velocity / (1 - (beta2 ** step))
                weights -= ((learning * momentum_cap) / (np.sqrt(velocity_cap + epsilon)))
                # important
                dw = 0

        error /= (2 * len(X))

        if (epoch + 1) % 50 == 0:
            print(f'Weights after epoch {epoch + 1}', weights)
            print(f'Error after epoch {epoch + 1}', error)


def adam_vanilla_gd(X, y, weights, epochs=500, learning=0.01, epsilon=1e-8, beta1=0.9, beta2=0.999, batch=2):
    # new
    momentum = 0
    velocity = 0
    for epoch in range(epochs):
        dw = 0
        error = 0
        for Xi, yi in zip(X, y):
            y_pred = predict(Xi, weights)
            dw += calculate_derivate(Xi, yi, y_pred)
            error += calculate_error(yi, y_pred)

        # note the additions - momentum and  velocity
        momentum = (beta1 * momentum) + ((1 - beta1) * dw)
        velocity = (beta2 * velocity) + ((1 - beta2) * (dw ** 2))
        step = epoch + 1
        momentum_cap = momentum / (1 - (beta1 ** step))
        velocity_cap = velocity / (1 - (beta2 ** step))
        weights -= ((learning * momentum_cap) / (np.sqrt(velocity_cap + epsilon)))

        error /= (2 * len(X))

        if (epoch + 1) % 50 == 0:
            print(f'Weights after epoch {epoch + 1}', weights)
            print(f'Error after epoch {epoch + 1}', error)


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

weights = np.array([-1, 1], dtype="longdouble")
adam_vanilla_gd(X, y, weights)