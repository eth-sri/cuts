import itertools
import torch
import numpy as np
from query import query_marginal


def expectation(data, dataset, features, function=lambda *x: x, condition=None, max_slice=1000):
    """
    Calculates the expectation of a function over the features on the current empirical distribution.

    :param data: (torch.tensor) One-hot encoded data.
    :param dataset: (BaseDataset) Instantiated dataset.
    :param features: (list) List of feature names over which the function applies and we calculate the expectation.
    :param function: (callable) The function we calculate the expectation of.
    :param condition: (torch.tensor) A binary vector of length len(data). This binary vector is produced indepently
        of this function, and is supposed to contain 1. in each location where the corresponding line in data fulfills
        some logical condition.
    :param max_slice: (int) Maximum slice for memory when calculating the marginal from the dataset.
    :return: (torch.float) The expected value of the function under the current empirical distribution.
    """
    # default: no conditioning
    if condition is None:
        condition = torch.ones(data.size()[0], device=data.device)
    
    # condition the data --> by masking replace all lines with zeros where the condition does not apply
    conditioned_data = condition.view((-1, 1)) * data

    # query the data to get the relevant marginal for the expectation
    unnormalized_marginal_probabilities = query_marginal(conditioned_data, tuple(features), dataset.full_one_hot_index_map, 
                                                         normalize=False, input_torch=True, max_slice=max_slice)
    normalization_constant = unnormalized_marginal_probabilities.sum()

    # evaluate the function for all possible values
    ranges = []
    for feature in features:
        if feature in dataset.discrete_features:
            ranges.append(np.arange(len(dataset.full_one_hot_index_map[feature])))
        else:
            ranges.append(dataset.train_bucketing_lower_edges[feature])
    function_values = torch.tensor([function(*arg_comb) for arg_comb in itertools.product(*ranges)]).to(data.device).float()
    Ef = (unnormalized_marginal_probabilities.flatten() * function_values.flatten()).sum() / normalization_constant

    return Ef

# TODO: the identity lambda function does not work for some reason as in the default
def variance(data, dataset, features, function=lambda *x: x, condition=None, max_slice=1000):
    """
    Calculates the variance of a function over the features on the current empirical distribution.

    :param data: (torch.tensor) One-hot encoded data.
    :param dataset: (BaseDataset) Instantiated dataset.
    :param features: (list) List of feature names over which the function applies and we calculate the variance.
    :param function: (callable) The function we calculate the variance of.
    :param condition: (torch.tensor) A binary vector of length len(data). This binary vector is produced indepently
        of this function, and is supposed to contain 1. in each location where the corresponding line in data fulfills
        some logical condition.
    :param max_slice: (int) Maximum slice for memory when calculating the marginal from the dataset.
    :return: (torch.float) The variance of the function under the current empirical distribution.
    """
    # default: no conditioning
    if condition is None:
        condition = torch.ones(data.size()[0], device=data.device)

    E_squared_of_f = expectation(data, dataset, features, function, condition, max_slice).pow(2)
    E_of_f_squared = expectation(data, dataset, features, lambda *x: function(*x)**2, condition, max_slice)

    Varf = E_of_f_squared - E_squared_of_f

    return Varf


def standard_deviation(data, dataset, features, function=lambda *x: x, condition=None, max_slice=1000):
    """
    Calculates the standard deviation of a function over the features on the current empirical distribution.

    :param data: (torch.tensor) One-hot encoded data.
    :param dataset: (BaseDataset) Instantiated dataset.
    :param features: (list) List of feature names over which the function applies and we calculate the standard deviation.
    :param function: (callable) The function we calculate the standard deviation of.
    :param condition: (torch.tensor) A binary vector of length len(data). This binary vector is produced indepently
        of this function, and is supposed to contain 1. in each location where the corresponding line in data fulfills
        some logical condition.
    :param max_slice: (int) Maximum slice for memory when calculating the marginal from the dataset.
    :return: (torch.float) The standard deviation of the function under the current empirical distribution.
    """
    # default: no conditioning
    if condition is None:
        condition = torch.ones(data.size()[0], device=data.device)

    std = variance(data, dataset, features, function, condition, max_slice).sqrt()
    
    return std


def entropy(data, dataset, features, function=None, condition=None, max_slice=1000):
    """
    Calculates the entropy of the features on the current empirical distribution.

    :param data: (torch.tensor) One-hot encoded data.
    :param dataset: (BaseDataset) Instantiated dataset.
    :param features: (list) List of feature names over which we calculate the entropy.
    :param function: (callable) Not used, there to match the signature of other operations.
    :param condition: (torch.tensor) A binary vector of length len(data). This binary vector is produced indepently
        of this function, and is supposed to contain 1. in each location where the corresponding line in data fulfills
        some logical condition.
    :param max_slice: (int) Maximum slice for memory when calculating the marginal from the dataset.
    :return: (torch.float) The entropy of the features under the current empirical distribution.
    """
    # default: no conditioning
    if condition is None:
        condition = torch.ones(data.size()[0], device=data.device)

    # condition the data --> by masking replace all lines with zeros where the condition does not apply
    conditioned_data = condition.view((-1, 1)) * data
    
    # query the data to get the relevant marginal
    marginal_probabilities = query_marginal(conditioned_data, tuple(features), dataset.full_one_hot_index_map, normalize=True, input_torch=True, max_slice=max_slice)

    H = - (marginal_probabilities * torch.log(marginal_probabilities + 1e-7)).sum()

    return H
