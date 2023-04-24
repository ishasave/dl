import numpy as np

def max_pooling_2d(x, pool_size, stride, padding):
    """
    Performs 2D max-pooling on a matrix x with the specified pool_size, stride, and padding.

    Arguments:
    x -- input matrix of shape (H, W)
    pool_size -- integer, size of the pooling window (usually square)
    stride -- integer, stride of the pooling operation
    padding -- integer, amount of zero padding to apply around the border of the matrix

    Returns:
    output -- matrix of shape (H_out, W_out)
    """
    # Determine the dimensions of the input matrix
    H, W = x.shape

    # Determine the dimensions of the output matrix
    H_out = int((H + 2 * padding - pool_size) / stride) + 1
    W_out = int((W + 2 * padding - pool_size) / stride) + 1

    # Apply padding to the input matrix
    x_padded = np.pad(x, padding, mode='constant')

    # Initialize the output matrix
    output = np.zeros((H_out, W_out))

    # Perform max-pooling
    for h in range(H_out):
        for w in range(W_out):
            h_start = h * stride
            h_end = h_start + pool_size
            w_start = w * stride
            w_end = w_start + pool_size
            pool_region = x_padded[h_start:h_end, w_start:w_end]
            output[h, w] = np.max(pool_region)

    return output
x = np.random.rand(5, 5) # example input matrix
pool_size = 2
stride = 2
padding = 1

output = max_pooling_2d(x, pool_size, stride, padding)
print(output)
