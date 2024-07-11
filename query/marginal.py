import numpy as np
import torch
from itertools import combinations


def query_marginal(data, marginal, one_hot_index_map, normalize=False, input_torch=None, max_slice=1000, max_memory=None, 
                   max_out_memory=None):
    """
    Function that calculates the empirical n-way marginal of the given features from the data.

    :param data: (np.ndarray or torch.tensor) The raw data one-hot encoded in all features.
    :param marginal: (list or tuple) The features in the desired marginal.
    :param one_hot_index_map: (dict) A dictionary containing the feature names and their index maps in the one-hot
        encoded data.
    :param normalize: (bool) Toggle to return the normalized marginal.
    :param input_torch: LEGACY ARGUMENT
    :param max_slice: (int) Partition the dataset in length into these slices to make sure that we fit in memory.
    :param max_memory: LEGACY ARGUMENT 
    :param max_out_memory: LEGACY ARGUMENT
    :return: (torch.tensor) The len(marginal)-way marginal of the features in marginal.
    """
    n_partitions = np.ceil(len(data) / max_slice).astype(int)
    return_marg = None
    
    for i in range(n_partitions):
        
        l, u = i * max_slice, min((i+1) * max_slice, len(data))
    
        if len(marginal) == 1:
            marg = data[l:u, one_hot_index_map[marginal[0]]]
        else:
            marg = data[l:u, one_hot_index_map[marginal[0]]]
            for m in marginal[1:]:
                reshape_to = np.ones(len(marg.size()) + 1).astype(int)
                reshape_to[0], reshape_to[-1] = u-l, len(one_hot_index_map[m])
                marg = data[l:u, one_hot_index_map[m]].view(tuple(reshape_to)) * marg.unsqueeze(-1)
        
        if return_marg is None:
            return_marg = marg.sum(0)
        else:
            return_marg += marg.sum(0)
        
        torch.cuda.empty_cache()
    
    if normalize:
        return return_marg / len(data)
    else:
        return return_marg


def get_all_marginals(features, degree, downward_closure=False):
    """
    Given a feature set, it returns the combinations of all degree-way marginals. If the 'downward_closure' parameter is
    set to True, we also return all lower-way marginals.

    :param features: (list) List containing all the feature names of the dataset.
    :param degree: (int) The degree of the marginals we want to get.
    :param downward_closure: (bool) If set to True, we also return all corresponding lower-degree marginals.
    :return: (list) The desired marginals in a list of tuples.
    """
    all_marginals = []
    for eff_deg in range(degree):
        deg = degree - eff_deg
        all_marginals.extend(list(combinations(features, deg)))
        if not downward_closure:
            break
    return all_marginals
