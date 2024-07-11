import torch


def gaussian_mechanism(q, sigma):
    """
    Takes a query q and the desired standard deviation ana returns the Gaussian perturbed answer. It will give a
    (sensitivity)^2/(2*sigma^2)-zCDP guarantee, where we mean the L2 sensitivity of the query.

    :param q: (torch.tensor) The true query answer to be perturbed.
    :param sigma: (float) The standard deviation of the noise, note that this defines the guarantee we get depending on
        the inherent L2 sensitivity of the query.
    :return: (torch.tensor) The Gaussian perturbed zCDP-guaranteed query answer.
    """
    mean = torch.zeros_like(q)
    std = sigma * torch.ones_like(q)
    return q + torch.normal(mean=mean, std=std)


def laplace_mechanism(q, scale):
    """
    Takes a query q and a desired scale and returns the Laplace perturbed query answer. The guarantee will be of
    (sensitivity/scale)-DP, where we mean the L1 sensitivity of the query.

    :param q: (torch.tensor) The true query answer to be perturbed.
    :param scale: (float) The scale of the Laplace distribution by which we perturb.
    :return: (torch.tensor) The Laplace perturbed query answer.
    """
    loc = torch.zeros_like(q)
    lap_scale = scale * torch.ones_like(scale)
    return torch.distributions.laplace.Laplace(loc=loc, scale=lap_scale).sample()


def exponential_mechanism(scores, epsilon, sensitivity):
    """
    Takes a finite vector of scores, and epsilon value, and the sensitivity of the scores to return a sample from the
    corresponding exponential-mechanism. The generated choice will be epsilon-DP.

    :param scores: (torch.tensor) Vector of scores that represent the quality of each possible choice.
    :param epsilon: (float) The epsilon value for the guarantee.
    :param sensitivity: (float) The sensitivity of the score function used to generate the scores.
    :return: (torch.float) Sample from the possible choice's corresponding Gibbs distribution with epsilon-DP guarantee.
    """
    p = torch.nn.functional.softmax(scores * epsilon / (2. * sensitivity), dim=-1)
    return torch.distributions.categorical.Categorical(probs=p).sample()
