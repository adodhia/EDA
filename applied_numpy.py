import numpy as np


def build_sequences(min_value, max_value, sequence_number):
    """
    Write a function that can generate the following sequences:
        sequence #1: 2 * n + 1
        sequence #2: 50 - 5 * n
        sequence #3: 2 ** n

    Although this exercises can easily be done with list
    comprehensions, it can be more efficient to use numpy
    (the arange method can be handy here).

    Start by generating all 50 first values for the sequence that
    was selected by sequence_number and return a numpy array
    filtered so that it only contains values in
    [min_value, max_value] (min and max being included)

    :param min_value: minimum value to use to filter the arrays
    :param max_value: maximum value to use to filter the arrays
    :param sequence_number: number of the sequence to return
    :returns: the right sequence as a np.array
    """

    if sequence_number == 1:
        seq = 2 * np.arange(50) + 1
    elif sequence_number == 2:
        seq = 50 - 5 * np.arange(50)
    elif sequence_number == 3:
        seq = 2**np.arange(50)
    else:
        raise ValueError

    return seq[(seq <= max_value) & (seq >= min_value)]



def moving_averages(x, k):
    """
    Given a numpy vector x of n > k, compute the moving averages
    of length k.  In other words, return a vector z of length
    m = n - k + 1 where z_i = mean([x_i, x_i-1, ..., x_i-k+1])

    Note that z_i refers to value of z computed from index i
    of x, but not z index i. z will be shifted compared to x
    since it cannot be computed for the first k-1 values of x.

    Example inputs:
    - x = [1, 2, 3, 4]
    - k = 3

    the moving average of 3 is only defined for the last 2
    values: [3, 4].
    And z = np.array([mean([1,2,3]), mean([2,3,4])])
        z = np.array([2.0, 3.0])

    :param x: numpy array of dimension n > k
    :param k: length of the moving average
    :returns: a numpy array z containing the moving averages.
    """

    n = x.shape[0]

    return np.array([np.mean(x[i - k:i]) for i in range(k, n + 1)])



def block_matrix(A, B):
    """
    Given two numpy matrices A and B of arbitrary dimensions,
    return a new numpy matrix of the following form:
        [A,0]
        [0,B]

    Example inputs:
        A = [1,2]    B = [5,6]
            [3,4]        [7,8]

    Expected output:
        [1,2,0,0]
        [3,4,0,0]
        [0,0,5,6]
        [0,0,7,8]

    :param A: numpy array
    :param B: numpy array
    :returns: a numpy array with A and B on the diagonal.
    """

    n = A.shape[0] + B.shape[0]
    p = A.shape[1] + B.shape[1]

    mat = np.zeros((n, p))
    mat[:A.shape[0], :A.shape[1]] = A
    mat[A.shape[0]:, A.shape[1]:] = B

    return mat


