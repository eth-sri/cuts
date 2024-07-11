import numpy as np
from query import get_all_marginals, query_marginal
from .marginal_error_functions import mean_squared_error, mean_absolute_error, kl_divergence, jensen_shannon_divergence


def evaluate_marginals(synthetic_data, real_data, dataset, errors='squared_error', marginals=2, return_mode='summary', verbose=False):
    """
    A function that evaluates the marginals of synthetic data with respect to the source real data it is meant to
    represent.

    :param synthetic_data: (np.ndarray) The fully one-hot encoded synthetic data as a numpy array.
    :param real_data: (np.ndarray) The fully one-hot encoded true data as a numpy array.
    :param dataset: (BaseDataset) The corresponding dataset object.
    :param errors: (str or list) The error(s) to evaluate. If it is a list, it should contain all errors to evaluate,
        if it is a single string, it has to mark the error we wish to evaluate. Available options are:
            - 'mean_squared_error': the MSE between the normalized histograms
            - 'mean_absolute_error': the mean L1 error between the normalized histograms
            - 'kl_divergence': the forward KL divergence with respect to the real data
            - 'jensen_shannon_divergence': the Jensen-Shannon divergence between the normalized histograms
    :param marginals: (int or list) The marginals we evaluate. If an integer is given, we evaluate all marginals of this
        dimension. Else, if a list is given, we evaluate all marginals contained in the list.
    :param return_mode: (str) If this is set to 'summary' we return for each error only the summary statistics over all
        evaluated marginals, in the following order: mean, std, median, min, max. In any other case, we return the
        errors for all marginals.
    :param verbose: (bool) If set to True, we provide progress information.
    :return: (dict) We return a dictionary with each evaluated error and the corresponding calculated error in the
        desired format. Additionally, when we return all errors, we also include the list of evaluated marginals in the
        dictionary, under the key 'evaluated_marginals'.
    """

    error_function = {
        'mean_squared_error': mean_squared_error,
        'mean_absolute_error': mean_absolute_error,
        'kl_divergence': kl_divergence,
        'jensen_shannon_divergence': jensen_shannon_divergence
    }

    # first prepare the list of marginals to evaluate
    if isinstance(marginals, int):
        marginals_to_evaluate = get_all_marginals(list(dataset.features.keys()), marginals, downward_closure=False)
    else:
        assert isinstance(marginals, list), 'The marginals argument has to be either an integer, meaning all ' \
                                            'marginals of this degree will be evaluated, or a list already ' \
                                            'containing the marginals to be evaluated against'
        marginals_to_evaluate = marginals
    
    if verbose:
        print('Querying Marginals')
    # now make the query both on the real data and the synthetic data + normalize them
    synthetic_marginals = [query_marginal(synthetic_data, m, dataset.full_one_hot_index_map, input_torch=False) / len(synthetic_data) for m in marginals_to_evaluate]
    if verbose:
        print('Synthetic Marginals Queried')
    real_marginals = [query_marginal(real_data, m, dataset.full_one_hot_index_map, input_torch=False) / len(real_data) for m in marginals_to_evaluate]
    if verbose:
        print('Real Marginals Queried')
        print('Calculating the Errors')

    # calculate the resulting error
    errors_to_calculate = [errors] if isinstance(errors, str) else errors
    return_dict = {}
    for error in errors_to_calculate:
        if verbose:
            print(f'Calculating {error}')
        all_errors = [error_function[error](p, q) for p, q in zip(real_marginals, synthetic_marginals)]
        if return_mode == 'summary':
            return_dict[error] = [np.mean(all_errors), np.std(all_errors), np.median(all_errors), np.min(all_errors), np.max(all_errors)]
        else:
            return_dict[error] = all_errors
    if verbose:
        print('Done')
    if not return_mode == 'summary':
        return_dict['evaluated_marginals'] = marginals_to_evaluate

    return return_dict
