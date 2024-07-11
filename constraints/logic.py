import torch
import numpy as np


def create_mask_from_parsed(data, dataset, feature, operator, constant):
    """
    Takes a feature from the dataset, a comparator, and a constant to which this feature is to be compared to, and
    returns the binary mask over the dataset that allows us to select all rows that fulfill the comparison.

    :param data: (torch.tensor) The full one hot encoded data.
    :param dataset: (BaseDataset) The instantiated base dataset.
    :param feature: (str) Feature in the data that is to be compared.
    :param operator: (str) The comparison operator.
    :param constant: (str) The constant to which the feature is compared.
    :return: (torhc.tensor) The binary mask of size data.size()[1].
    """
    # check if the feature exists
    assert feature in list(dataset.features.keys()), f'{feature} is not a feature of {dataset.name}'
    
    # continuous features
    if feature in dataset.continuous_features:

        feature_idx = np.argwhere(np.array(dataset.continuous_features) == feature).item()

        # make sure that we support the operator
        admissible_operators = ['<', '>', '>=', '<=', '==', '!=']
        if operator not in admissible_operators:
            raise ValueError(f'Operator {operator} is invalid, supported operators for continuous features are: {admissible_operators}')
        
        # check where this constant would fall when discretized the same way as the feature itself
        # TODO: not sure if having this as an integer here makes sense
        step = (dataset.maxs[feature_idx] + 1e-7 - dataset.mins[feature_idx]) / dataset.discretization_one_hot_train
        constant_discretized = min(max(0, np.floor((float(constant) - dataset.mins[feature_idx]) / step).astype(int)), dataset.discretization_one_hot_train - 1)

        # create the mask
        mask = torch.zeros(dataset.discretization_one_hot_train).to(data.device)
        
        if operator == ">":
            if not constant_discretized + 1 >= dataset.discretization_one_hot_train:
                mask[constant_discretized+1:] = 1.
        
        elif operator == "<=":
            mask[:constant_discretized+1] = 1.
            
        elif operator == ">=":
            mask[constant_discretized:] = 1.
        
        elif operator == '<':
            mask[:constant_discretized] = 1.
        
        elif operator == '==':
            mask[constant_discretized] = 1.
        
        elif operator == '!=':
            mask[constant_discretized] = 1.
            mask = 1. - mask

    # discrete features
    else:

        # make sure that this is a valid constant
        assert constant in dataset.features[feature], f'{constant} is not in the domain of {feature}'

        # make sure that we support the operator
        admissible_operators = ['==', '!=']
        if operator not in admissible_operators:
            raise ValueError(f'Operator {operator} is invalid, supported operators for discrete features are: {admissible_operators}')
        
        constant_idx = np.argwhere(np.array(dataset.features[feature]) == constant).item()
        
        mask = torch.zeros(len(dataset.features[feature])).to(data.device)

        if operator == '==':
            mask[constant_idx] = 1.
        elif operator == '!=':
            mask[constant_idx] = 1.
            mask = 1. - mask
        
    # prepare the global full mask on the whole dataset
    full_mask = torch.zeros(data.size()[1]).to(data.device)
    full_mask[dataset.full_one_hot_index_map[feature]] = mask

    return full_mask


def create_mask(data, dataset, condition):
    """
    Takes a condition of either of the two forms:
        - feature_name=feature_value,
        - feature_name=NOT feature_value,
    and return the corresponding binary mask that allows us to mark the rows where the condition applies.

    :param data: (torch.tensor) The one-hot encoded dataset.
    :param dataset: (BaseDataset) The corresponding instantiated dataset.
    :param condition: (str) The condition as a string in the form explained above.
    :return: (torch.tensor) The resulting binary mask.
    """
    cond_feature_name, cond_feature_value = condition.split('=')

    # check if the condition was negated
    negated = False
    if cond_feature_value.startswith('NOT '):
        cond_feature_value = cond_feature_value[4:]
        negated = True

    # create the mask
    cond_loc = np.argwhere(np.array(dataset.features[cond_feature_name]) == cond_feature_value).item()
    mask = torch.zeros(data.size()[1]).to(data.device)
    if negated:
        mask[dataset.full_one_hot_index_map[cond_feature_name]] = 1. 
    mask[dataset.full_one_hot_index_map[cond_feature_name][0] + cond_loc] = 0. if negated else 1.

    return mask


def and_gate(data, dataset, condition_A, condition_B):
    """
    A AND B

    Takes two conditions of the form feature_name=feature_value or feature_name=NOT feature_value, and returns a vector
    of the same length of that of 'data' with 0 at all indexes in which lines of 'data' the logical AND of the two
    conditions is not met, and with 1s marking all lines where the logical AND of the two conditions is satisfied.

    :param data: (torch.tensor) The one-hot encoded dataset.
    :param dataset: (BaseDataset) The corresponding instantiated dataset.
    :param condition_A: (str) The first condition to the AND operation, in the format explained above.
    :param condition_B: (str) The second condition to the AND operation, in the format explained above.
    :return: (torch.tensor) A binary vector of length len(data), each on-zero position marking a line 'data' where
        A AND B is satisfied.
    """
    m_A = create_mask(data, dataset, condition_A)
    m_B = create_mask(data, dataset, condition_B)
    fulfilled_rows = (data @ m_A.T) * (data @ m_B.T)
    return fulfilled_rows


def or_gate(data, dataset, condition_A, condition_B, true_or=True):
    """
    A OR B

    Takes two conditions of the form feature_name=feature_value or feature_name=NOT feature_value, and returns a vector
    of the same length of that of 'data' with 0 at all indexes in which lines of 'data' the logical OR of the two
    conditions is not met, with 1s marking all lines where the logical OR of the two conditions is satisfied due to
    exactly one of A or B being met, and if in the given line both A and B are met the line is marked by 2, unless
    'true_or' is set to True, in which case we clamp to 1.

    :param data: (torch.tensor) The one-hot encoded dataset.
    :param dataset: (BaseDataset) The corresponding instantiated dataset.
    :param condition_A: (str) The first condition to the AND operation, in the format explained above.
    :param condition_B: (str) The second condition to the AND operation, in the format explained above.
    :param true_or: (bool) Toggle to make sure to return a binary vector by clamping to 1.
    :return: (torch.tensor) A vector of length len(data), each on-zero position marking a line 'data' where A OR B is
        satisfied. Note that if 'true_or' is set to False,
    """
    m_A = create_mask(data, dataset, condition_A)
    m_B = create_mask(data, dataset, condition_B)
    fulfilled_rows = data @ (m_A.T + m_B.T)
    if true_or:
        clamped_rows = torch.clamp(fulfilled_rows, max=1.0)
        return clamped_rows
    return fulfilled_rows


def implication_violation(data, dataset, condition_A, condition_B):
    """
    A => B not satisfied

    Takes two conditions A and B of the form feature_name=feature_value or feature_name=NOT feature_value, and returns
    a binary vector of length len(data) where all non-zero elements correspond to a violation of A => B, which is
    equivalent to A AND NOT B being satisfied.

    :param data: (torch.tensor) The one-hot encoded dataset.
    :param dataset: (BaseDataset) The corresponding instantiated dataset.
    :param condition_A: (str) The first condition to the AND operation, in the format explained above.
    :param condition_B: (str) The second condition to the AND operation, in the format explained above.
    :return: (torch.tensor) All rows where a violation is present marked by 1, else by 0.
    """
    # note that violations to A => B are detected in all rows where A AND (NOT B) is satisfied
    # negate B
    cond_B_feature_name, cond_B_feature_value = condition_B.split('=')
    not_condition_B = cond_B_feature_name + '=NOT ' + cond_B_feature_value
    violating_rows = and_gate(data, dataset, condition_A, not_condition_B)
    return violating_rows
