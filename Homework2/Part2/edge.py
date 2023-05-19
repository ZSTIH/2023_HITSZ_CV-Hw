import numpy as np

def conv(image, kernel):
    """ An implementation of convolution filter.

    This function uses element-wise multiplication and np.sum()
    to efficiently compute weighted sum of neighborhood at each
    pixel.

    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk).

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    Hi, Wi = image.shape
    Hk, Wk = kernel.shape
    out = np.zeros((Hi, Wi))

    # For this assignment, we will use edge values to pad the images.
    # Zero padding will make derivatives at the image boundary very big,
    # whereas we want to ignore the edges at the boundary.
    pad_width0 = Hk // 2
    pad_width1 = Wk // 2
    pad_width = ((pad_width0,pad_width0),(pad_width1,pad_width1))
    padded = np.pad(image, pad_width, mode='edge')

    ### YOUR CODE HERE
    pass
    kernel = np.flip(np.flip(kernel, axis=0), axis=1)
    for row in range(Hi):
        for col in range(Wi):
            out[row, col] = np.sum(kernel * padded[row : row + Hk, col : col + Wk])
    ### END YOUR CODE

    return out

def gaussian_kernel(size, sigma):
    """ Implementation of Gaussian Kernel.

    This function follows the gaussian kernel formula,
    and creates a kernel matrix.

    Hints:
    - Use np.pi and np.exp to compute pi and exp.

    Args:
        size: int of the size of output matrix.
        sigma: float of sigma to calculate kernel.

    Returns:
        kernel: numpy array of shape (size, size).
    """

    kernel = np.zeros((size, size))

    ### YOUR CODE HERE
    pass
    k = size // 2
    for i in range(size):
        for j in range(size):
            kernel_x = np.exp(- (i - k) ** 2 / 2 / sigma ** 2) / np.sqrt(2 * np.pi) / sigma
            kernel_y = np.exp(- (j - k) ** 2 / 2 / sigma ** 2) / np.sqrt(2 * np.pi) / sigma
            kernel[i, j] = kernel_x * kernel_y
    ### END YOUR CODE

    return kernel

def partial_x(img):
    """ Computes partial x-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: x-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    pass
    kernel = [0.5, 0, -0.5]
    kernel = np.array(kernel).reshape(1, 3)
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def partial_y(img):
    """ Computes partial y-derivative of input img.

    Hints:
        - You may use the conv function in defined in this file.

    Args:
        img: numpy array of shape (H, W).
    Returns:
        out: y-derivative image.
    """

    out = None

    ### YOUR CODE HERE
    pass
    kernel = [0.5, 0, -0.5]
    kernel = np.array(kernel).reshape(3, 1)
    out = conv(img, kernel)
    ### END YOUR CODE

    return out

def gradient(img):
    """ Returns gradient magnitude and direction of input img.

    Args:
        img: Grayscale image. Numpy array of shape (H, W).

    Returns:
        G: Magnitude of gradient at each pixel in img.
            Numpy array of shape (H, W).
        theta: Direction(in degrees, 0 <= theta < 360) of gradient
            at each pixel in img. Numpy array of shape (H, W).

    Hints:
        - Use np.sqrt and np.arctan2 to calculate square root and arctan
    """
    G = np.zeros(img.shape)
    theta = np.zeros(img.shape)

    ### YOUR CODE HERE
    pass
    Gx = partial_x(img)
    Gy = partial_y(img)
    G = np.sqrt(Gx ** 2 + Gy ** 2)
    theta = (np.arctan2(Gy, Gx) * 180 / np.pi + 360) % 360
    ### END YOUR CODE

    return G, theta


def non_maximum_suppression(G, theta):
    """ Performs non-maximum suppression.

    This function performs non-maximum suppression along the direction
    of gradient (theta) on the gradient magnitude image (G).

    Args:
        G: gradient magnitude image with shape of (H, W).
        theta: direction of gradients with shape of (H, W).

    Returns:
        out: non-maxima suppressed image.
    """
    H, W = G.shape
    out = np.zeros((H, W))

    # Round the gradient direction to the nearest 45 degrees
    theta = np.floor((theta + 22.5) / 45) * 45
    theta = (theta % 360.0).astype(np.int32)

    #print(G)
    ### BEGIN YOUR CODE
    pass
    temp1 = 0
    temp2 = 0
    for row in range(H):
        for col in range(W):
            theta0 = theta[row, col]
            if theta0 == 0 or theta0 == 180:
                if col < W - 1:
                    temp1 = G[row, col + 1]
                else:
                    temp1 = 0
                if col > 0:
                    temp2 = G[row, col - 1]
                else:
                    temp2 = 0
            elif theta0 == 45 or theta0 == 225:
                if row < H - 1 and col < W - 1:
                    temp1 = G[row + 1, col + 1]
                else:
                    temp1 = 0
                if row > 0 and col > 0:
                    temp2 = G[row - 1, col - 1]
                else:
                    temp2 = 0
            elif theta0 == 90 or theta0 == 270:
                if row < H - 1:
                    temp1 = G[row + 1, col]
                else:
                    temp1 = 0
                if row > 0:
                    temp2 = G[row - 1, col]
                else:
                    temp2 = 0
            else:
                if row > 0 and col < W - 1:
                    temp1 = G[row - 1, col + 1]
                else:
                    temp1 = 0
                if row < H - 1 and col > 0:
                    temp2 = G[row + 1, col - 1]
                else:
                    temp2 = 0
            if G[row, col] >= max(temp1, temp2):
                out[row, col] = G[row, col]
            else:
                out[row, col] = 0
    ### END YOUR CODE

    return out

def double_thresholding(img, high, low):
    """
    Args:
        img: numpy array of shape (H, W) representing NMS edge response.
        high: high threshold(float) for strong edges.
        low: low threshold(float) for weak edges.

    Returns:
        strong_edges: Boolean array representing strong edges.
            Strong edeges are the pixels with the values greater than
            the higher threshold.
        weak_edges: Boolean array representing weak edges.
            Weak edges are the pixels with the values smaller or equal to the
            higher threshold and greater than the lower threshold.
    """

    strong_edges = np.zeros(img.shape, dtype=bool)
    weak_edges = np.zeros(img.shape, dtype=bool)

    ### YOUR CODE HERE
    pass
    strong_edges = img > high
    weak_edges = (img > low) & (img <= high)
    ### END YOUR CODE

    return strong_edges, weak_edges


def get_neighbors(y, x, H, W):
    """ Return indices of valid neighbors of (y, x).

    Return indices of all the valid neighbors of (y, x) in an array of
    shape (H, W). An index (i, j) of a valid neighbor should satisfy
    the following:
        1. i >= 0 and i < H
        2. j >= 0 and j < W
        3. (i, j) != (y, x)

    Args:
        y, x: location of the pixel.
        H, W: size of the image.
    Returns:
        neighbors: list of indices of neighboring pixels [(i, j)].
    """
    neighbors = []

    for i in (y-1, y, y+1):
        for j in (x-1, x, x+1):
            if i >= 0 and i < H and j >= 0 and j < W:
                if (i == y and j == x):
                    continue
                neighbors.append((i, j))

    return neighbors

def link_edges(strong_edges, weak_edges):
    """ Find weak edges connected to strong edges and link them.

    Iterate over each pixel in strong_edges and perform breadth first
    search across the connected pixels in weak_edges to link them.
    Here we consider a pixel (a, b) is connected to a pixel (c, d)
    if (a, b) is one of the eight neighboring pixels of (c, d).

    Args:
        strong_edges: binary image of shape (H, W).
        weak_edges: binary image of shape (H, W).
    
    Returns:
        edges: numpy boolean array of shape(H, W).
    """

    H, W = strong_edges.shape
    indices = np.stack(np.nonzero(strong_edges)).T
    edges = np.zeros((H, W), dtype=bool)

    # Make new instances of arguments to leave the original
    # references intact
    weak_edges = np.copy(weak_edges)
    edges = np.copy(strong_edges)

    ### YOUR CODE HERE
    pass
    flag = True
    while flag:
        flag = False
        for indice in indices:
            neighbours = get_neighbors(indice[0], indice[1], H, W)
            for neighbour in neighbours:
                if weak_edges[neighbour] and not edges[neighbour]:
                    flag = True
                    edges[neighbour] = True
        indices = np.stack(np.nonzero(edges)).T
    ### END YOUR CODE

    return edges

def canny(img, kernel_size=5, sigma=1.4, high=20, low=15):
    """ Implement canny edge detector by calling functions above.

    Args:
        img: binary image of shape (H, W).
        kernel_size: int of size for kernel matrix.
        sigma: float for calculating kernel.
        high: high threshold for strong edges.
        low: low threashold for weak edges.
    Returns:
        edge: numpy array of shape(H, W).
    """
    ### YOUR CODE HERE
    pass
    kernel = gaussian_kernel(kernel_size, sigma)
    new_img = conv(img, kernel)
    G, theta = gradient(new_img)
    nms_img = non_maximum_suppression(G, theta)
    strong_edges, weak_edges = double_thresholding(nms_img, high, low)
    edge = link_edges(strong_edges, weak_edges)
    ### END YOUR CODE

    return edge
