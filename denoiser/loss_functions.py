import torch


def total_variation_loss(p, q):
    """
    Calculates the normalized total variation distance between two discrete probability distributions over the same set.

    :param p: (torch.tensor) Probability distribution 1.
    :param q: (torch.tensor) Probability distribution 2.
    :return: (torch.tensor) The total variation distance between the two distributions.
    """
    loss = 0.5 * (p - q).abs().mean()
    return loss


def mean_squared_error_loss(p, q):
    """
    Calculates the mean squared error between two distributions, p and q.

    :param p: (torch.tensor) Probability distribution 1.
    :param q: (torch.tensor) Probability distribution 2.
    :return: (torch.tensor) The mean squared error between the two distributions.
    """
    loss = (p - q).pow(2).mean()
    return loss


def kl_divergence(p, q, num_stable=True):
    """
    Calculates the KL-divergence between two discrete probability distributions over the same set.

    :param p: (torch.tensor) Probability distribution 1.
    :param q: (torch.tensor) Probability distribution 2.
    :param num_stable: (bool) Toggle to add some constant in the log for numeric stability.
    :return: (torch.tensor) The KL-divergence between the two distributions.
    """
    if num_stable:
        loss = (p * torch.log(torch.div(p + 1e-4, q + 1e-4))).sum()
    else:
        loss = (p * torch.log(torch.div(p, q))).sum()
    return loss


def jensen_shannon_divergence(p, q):
    """
    Calculates the Jensen-Shannon-divergence between two discrete probability distributions over the same set.

    :param p: (torch.tensor) Probability distribution 1.
    :param q: (torch.tensor) Probability distribution 2.
    :return: (torch.tensor) The Jensen-Shannon divergence between the two distributions.
    """
    m = 0.5 * (p + q)
    loss = 0.5 * (kl_divergence(p, m, num_stable=True) + kl_divergence(q, m, num_stable=True))
    return loss
