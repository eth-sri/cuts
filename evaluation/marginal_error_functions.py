import numpy as np


def mean_squared_error(p, q):
    """
    Calculates the mean squared error between two (normalized) histograms.

    :param p: (np.ndarray) Normalized histogram 1.
    :param q: (np.ndarray) Normalized histogram 2.
    :return: (np.float) Mean squared error of the two histograms.
    """
    return np.mean((p - q)**2)


def mean_absolute_error(p, q):
    """
    Calculates the mean absolute error between two (normalized) histograms.

    :param p: (np.ndarray) Normalized histogram 1.
    :param q: (np.ndarray) Normalized histogram 2.
    :return: (np.float) Mean absolute error of the two histograms.
    """
    return np.mean(np.abs(p - q))


def kl_divergence(p, q):
    """
    Calculates the forward KL-divergence of two normalized histograms p and q.

    :param p: (np.ndarray) The forward distribution.
    :param q: (np.ndarray) The 'guessed' distribution.
    :return: (np.float) The forward KL-divergence KL(p||q).
    """
    return np.sum(p * np.log((p + 1e-6)/(q + 1e-6)))


def jensen_shannon_divergence(p, q):
    """
    Calculates the Jensen-Shannon divergence between two normalized histograms p and q.

    :param p: (np.ndarray) Normalized histogram 1.
    :param q: (np.ndarray) Normalized histogram 2.
    :return: (np.float) The Jensen-Shannon divergence between p and q.
    """
    m = (p + q) / 2
    return (kl_divergence(p, m) + kl_divergence(q, m)) / 2
