import numpy as np

def conv2d(x, kernel, padding):
    """
    Performs 2D convolution between an input matrix x and a filter kernel with specified padding.

    Arguments:
    x -- input matrix of shape (H, W)
    kernel -- filter kernel of shape (M, M)
    padding -- integer, amount of zero padding to apply around the border of the matrix

    Returns:
    output -- matrix of shape (H_out, W_out)
    """
    # Determine the dimensions of the input matrix and filter kernel
    H, W = x.shape
    M, _ = kernel.shape

    # Determine the dimensions of the output matrix
    H_out = H + 2 * padding - M + 1
    W_out = W + 2 * padding - M + 1

    # Apply padding to the input matrix
    x_padded = np.pad(x, padding, mode='constant')

    # Initialize the output matrix
    output = np.zeros((H_out, W_out))

    # Perform convolution
    for h in range(H_out):
        for w in range(W_out):
            h_start = h
            h_end = h_start + M
            w_start = w
            w_end = w_start + M
            x_slice = x_padded[h_start:h_end, w_start:w_end]
            output[h, w] = np.sum(x_slice * kernel)

    return output
x = np.random.rand(5, 5) # example input matrix
kernel = np.random.rand(3, 3) # example filter kernel
padding = 1

output = conv2d(x, kernel, padding)
print(output)