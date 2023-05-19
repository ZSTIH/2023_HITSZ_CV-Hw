import numpy as np


def conv_nested(image, kernel):
    """A naive implementation of convolution filter.

    This is a naive implementation of convolution using 4 nested for-loops.
    This function computes convolution of an image with a kernel and outputs
    the result that has the same shape as the input image.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    for h in range(Hi):
        for w in range(Wi):
            for i in range(Hk):
                for j in range(Wk):
                    row_image = h + 1 - i
                    col_image = w + 1 - j
                    if row_image >= 0 and col_image >= 0 and row_image < Hi and col_image < Wi:
                        out[h, w] += image[row_image, col_image] * kernel[i, j]
                    else:
                        out[h, w] +=0
    ### END YOUR CODE

    return out

def zero_pad(image, pad_height, pad_width):
    """ Zero-pad an image.

    Ex: a 1x1 image [[1]] with pad_height = 1, pad_width = 2 becomes:

        [[0, 0, 0, 0, 0],
         [0, 0, 1, 0, 0],
         [0, 0, 0, 0, 0]]         of shape (3, 5)

    Args:
        image: numpy array of shape (H, W).
        pad_width: width of the zero padding (left and right padding).
        pad_height: height of the zero padding (bottom and top padding).

    Returns:
        out: numpy array of shape (H+2*pad_height, W+2*pad_width).
    """

    H, W = image.shape
    out = None

    ### YOUR CODE HERE
    pass
    out = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), 'constant', constant_values=0)
    ### END YOUR CODE
    return out


def conv_fast(image, kernel):
    """ An efficient implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Hints:
        - Use the zero_pad function you implemented above
        - There should be two nested for-loops
        - You may find np.flip() and np.sum() useful

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    ### YOUR CODE HERE
    pass
    padding_h = Hk // 2
    padding_w = Wk // 2
    padding_image = zero_pad(image, padding_h, padding_w)
    filpped_kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    for h in range(Hi):
        for w in range(Wi):
            out[h, w] = np.sum(padding_image[h : h + Hk, w : w + Wk] * filpped_kernel)
    ### END YOUR CODE

    return out

def cross_correlation(f, g):
    """ Cross-correlation of image f and template g.

    Hint: use the conv_fast function defined above.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    pass
    out = conv_fast(f, np.flip(np.flip(g, axis=0), axis=1))
    ### END YOUR CODE

    return out

def zero_mean_cross_correlation(f, g):
    """ Zero-mean cross-correlation of image f and template g.

    Subtract the mean of g from g so that its mean becomes zero.

    Hint: you should look up useful numpy functions online for calculating the mean.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    pass
    new_g = g - np.mean(g)
    out = conv_fast(f, np.flip(np.flip(new_g, axis=0), axis=1))
    ### END YOUR CODE

    return out

def normalized_cross_correlation(f, g):
    """ Normalized cross-correlation of image f and template g.

    Normalize the subimage of f and the template g at each step
    before computing the weighted sum of the two.

    Hint: you should look up useful numpy functions online for calculating 
          the mean and standard deviation.

    Args:
        f: numpy array of shape (Hf, Wf).
        g: numpy array of shape (Hg, Wg).

    Returns:
        out: numpy array of shape (Hf, Wf).
    """

    out = None
    ### YOUR CODE HERE
    pass
    Hf, Wf = f.shape
    Hg, Wg = g.shape
    out = np.zeros((Hf, Wf))
    g_mean = np.mean(g)
    g_std = np.std(g)
    new_g = (g - g_mean) / g_std
    new_f = zero_pad(f, Hg // 2, Wg // 2)
    for row in range(Hf):
        for col in range(Wf):
            f_patch = new_f[row : row + Hg, col : col + Wg]
            f_patch_mean = np.mean(f_patch)
            f_patch_std = np.std(f_patch)
            new_f_patch = (f_patch - f_patch_mean) / f_patch_std
            out[row, col] = np.sum(new_g * new_f_patch)
    ### END YOUR CODE

    return out
