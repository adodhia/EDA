"""This file contains a set of functions to implement using PCA.
All of them take at least a dataframe df as argument. To test your functions
locally, we recommend using the wine dataset that you can load from sklearn by
importing sklearn.datasets.load_wine"""

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def get_cumulated_variance(df, scale):
    """Apply PCA on a DataFrame and return a new DataFrame containing
    the cumulated explained variance from using the first component only,
    up to using all components together. Values should be expressed as
    a percentage of the total variance explained.

    The DataFrame will have one row and each column should correspond to a
    principal component.

    Example:
             PC1        PC2        PC3        PC4    PC5
    0  36.198848  55.406338  66.529969  73.598999  100.0

    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param scale: boolean, whether to scale or not
    :return: a new DataFrame with cumulated variance in percent
    """

    if scale:
        scaler = StandardScaler()
        df = scaler.fit_transform(df)

    pca = PCA()
    pca.fit(df)

    cumsum = np.cumsum(pca.explained_variance_ratio_).reshape(1, -1)
    return pd.DataFrame(
        100 * cumsum,
        columns=["PC{}".format(n + 1) for n in range(df.shape[1])])



def get_coordinates_of_first_two(df, scale):
    """Apply PCA on a given DataFrame df and return a new DataFrame
    containing the coordinates of the two first principal components
    expressed in the original basis (with the original columns).

    Example:
    if the original DataFrame was:

          A    B
    0   1.3  1.2
    1    27  2.1
    2   3.3  6.8
    3   5.1  3.2

    we want the components PC1 and PC2 expressed as a linear combination
    of A and B, presented in a table as:

              A    B
    PC1     0.1  1.1
    PC2       3    1

    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param scale: boolean, whether to scale or not
    :return: a new DataFrame with coordinates of PC1 and PC2
    """

    cols = df.columns

    if scale:
        scaler = StandardScaler()
        df = scaler.fit_transform(df)

    pca = PCA()
    pca.fit(df)

    first_two = pca.components_[:2]
    return pd.DataFrame(first_two, columns=cols, index=["PC1", "PC2"])



def get_most_important_two(df, scale):
    """Apply PCA on a given DataFrame df and use it to determine the
    'most important' features in your dataset. To do so we will focus
    on the principal component that exhibits the highest explained
    variance (that's PC1).

    PC1 can be expressed as a vector in the original basis (our original
    columns). Here we want to return the names of the two features that
    have the highest absolute weight in PC1.

    Example:
        if the original DataFrame was:

          A    B    C
    0   1.3  1.2  0.1
    1    27  2.1  1.2
    2   3.3  6.8  3.4
    3   5.1  3.2  4.5

    and PC1 can be written as [.01, .9, .2] in [A, B, C].

    Then you should return B, C as the two most important features.

    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param scale: boolean, whether to scale or not
    :return: names of the two most important features as a tuple
    """

    raise NotImplementedError


def distance_in_n_dimensions(df, point_a, point_b, n, scale):
    """Write a function that applies PCA on a given DataFrame df in order to learn
    a new subspace of dimension n.

    Project the two points point_a and point_b into that n dimensions space,
    compute the Euclidean distance between the points in that space and return it.

    Example:
        if the original DataFrame was:

          A    B    C
    0   1.3  1.2  0.1
    1    27  2.1  1.2
    2   3.3  6.8  3.4
    3   5.1  3.2  4.5

    and n = 2, you can learn a new subspace with two columns [PC1, PC2].

    Then given two points:

    point_a = [1, 2, 3]
    point_b = [2, 3, 4]
    expressed in [A, B, C]

    Project them into [PC1, PC2] and return the Euclidean distance between the
    points in that space.

    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param point_a: a numpy vector expressed in the same basis as df
    :param point_b: a numpy vector expressed in the same basis as df
    :param n: number of dimensions of the new space
    :param scale: whether to scale data or not
    :return: distance between points in the subspace
    """

    raise NotImplementedError


def find_outliers_pca(df, n, scale):
    """Apply PCA on a given DataFrame df and project all the data
    on the first principal component.

    With all the points projected in a one-dimension space, find outliers
    by looking for points that lie at more than n standard deviations from the mean.

    You should return a new dataframe containing all the rows of the original dataset
    that have been found to be outliers when projected.

    Example:
        if the original DataFrame was:

          A    B    C
    0   1.3  1.2  0.1
    1    27  2.1  1.2
    2   3.3  6.8  3.4
    3   5.1  3.2  4.5

    Once projected on PC1 it will be:
        PC1
    0     1
    1   1.1
    2   2.1
    3   100

    Compute the mean of this one dimensional dataset and find all rows that lie at more
    than n standard deviations from it. Here only the row 3 is an outlier.

    So you should return:
          A    B    C
    3   5.1  3.2  4.5

    If scale is True, you should standardise the data first
    Tip: use the StandardScaler from sklearn

    :param df: pandas DataFrame
    :param n: number of standard deviations from the mean to be considered outlier
    :param scale: whether to scale data or not
    :return: pandas DataFrame containing outliers only
    """

    raise NotImplementedError


