import torch
import numpy as np
from copy import copy
from query import query_marginal


def demographic_parity_distance(data, target_feature, protected_feature, dataset, operation='mean'):
    """
    Calculates the demographic parity distance on the data with respect to a target feature and the given protected feature. 
    We allow for two options: mean, if we want the mean DP distance of all pairwise configurations, or max, if we want to
    have the maximal DP distance, as in the defintion. Note that the first option is more practical whenever this function
    is used in training where gradients may be computed, as max produces sparse gradients.

    :param data: (torch.tensor) The full one hot encoded data. Note that if you want to test a classifier for DP, then you
        first have to concatenate the predicted labels in a correct format to the data.
    :param target_feature: (str) The name of the target (label) feature.
    :param protected_feature: (str) The name of the protected feature.
    :param dataset: (BaseDataset) The instantiated dataset object containing the necessary information for the data.
    :param operation: (str) The operation applied to aggregate the absolute differences in the expected labeling. Available are
        only mean or max. Note that mean is preferable when we have to differentiate through this function.
    :return: (torch.float) The aggregated DP distance on the current constellation.
    """
    assert operation in ['mean', 'max'], 'Only mean and max operations are available'
    target_protected_marginal = query_marginal(data, (target_feature, protected_feature), dataset.full_one_hot_index_map, normalize=True, input_torch=True)
    # renormalize the marginal such that each column sums to 1
    normalization_constant = target_protected_marginal.sum(0)
    normalization_constant = torch.where(normalization_constant == 0, torch.tensor(1.0).to(data.device), normalization_constant)
    renormalized_target_protected_marginal = target_protected_marginal / normalization_constant.view((1, -1))
    expected_target_per_protected = (torch.arange(target_protected_marginal.size()[0], device=data.device).view((-1, 1)) * renormalized_target_protected_marginal).sum(0)
    all_differences = expected_target_per_protected.view((-1, 1)) - expected_target_per_protected.view((1, -1))
    op_of_absolute_differences = all_differences.abs().mean() if operation == 'mean' else all_differences.abs().max()
    return op_of_absolute_differences


def equalized_odds_distance(data, true_labels, target_feature, protected_feature, desired_outcome, dataset, operation='mean'):
    """
    Calculates the equalized odds distance with respect to a target feature and a protected feature. We allow for two options: 
    mean, if we want the mean EO distance of all pairwise configurations, or max, if we want to have the maximal EO distance, as 
    in the defintion. Note that the first option is more practical whenever this function is used in training where gradients may 
    be computed, as max produces sparse gradients. Note that this notion at the moment is only sensibly supported for binary targets.
    TODO: by a bit more care, this can easily be extended to more than just one class, by simply grouping all classes under the negative

    :param data: (torch.tensor) The full one hot encoded data. Note that if you want to test a classifier for DP, then you
        first have to concatenate the predicted labels in a correct format to the data.
    :param true_labels: (torch.tensor) The true reference labels, required to be used as the reference for calculating the
        TPR and the FPR.
    :param target_feature: (str) The name of the target (label) feature.
    :param protected_feature: (str) The name of the protected feature.
    :param desired_outcome: (str) The label of the desired outcome (positive) with respect to which we calculate the TPR and the FPR.
    :param dataset: (BaseDataset) The instantiated dataset object containing the necessary information for the data.
    :param operation: (str) The operation applied to aggregate the absolute differences in the expected labeling. Available are
        only mean or max. Note that mean is preferable when we have to differentiate through this function.
    :return: (torch.float) The aggregated violation of the equalized odds fairness metric.
    """
    assert operation in ['mean', 'max'], 'Only mean and max operations are available'
    desired_idx = np.argwhere(np.array(dataset.features[target_feature], dtype=str) == str(desired_outcome)).flatten().item()
    reference_target_feature = target_feature + '_ref'
    # add the true labels to the data
    extended_data = torch.cat((data, true_labels), dim=1)
    # extend the index_map
    full_one_hot_index_map = copy(dataset.full_one_hot_index_map)
    full_one_hot_index_map[reference_target_feature] = data.size()[1] + np.arange(true_labels.size()[1])

    target_protected_ref_marginal = query_marginal(
        extended_data, (target_feature, protected_feature, reference_target_feature), full_one_hot_index_map, normalize=False, input_torch=True
    )
    # renormalize to obtain TPR, FPR, TNR, FNR per protected feature
    normalization_constant = target_protected_ref_marginal.sum(0)
    target_protected_ref_marginal = target_protected_ref_marginal / normalization_constant.unsqueeze(0)

    # keep only the PR-s
    tpr_per_protected, fpr_per_protected = target_protected_ref_marginal[desired_idx, :, desired_idx], target_protected_ref_marginal[desired_idx, :, 1-desired_idx]
    tpr_diffs, fpr_diffs = tpr_per_protected.view((-1, 1)) - tpr_per_protected.view((1, -1)), fpr_per_protected.view((-1, 1)) - fpr_per_protected.view((1, -1))
    
    op_of_absolute_diffs_tpr = tpr_diffs.abs().mean() if operation == 'mean' else tpr_diffs.abs().max()
    op_of_absolute_diffs_fpr = fpr_diffs.abs().mean() if operation == 'mean' else fpr_diffs.abs().max()

    return 0.5 * (op_of_absolute_diffs_tpr + op_of_absolute_diffs_fpr)


def equality_of_opportunity_distance(data, true_labels, target_feature, protected_feature, desired_outcome, dataset, operation='mean'):
    """
    Calculates the equality of opportunity distance with respect to a target feature and a protected feature. We allow for two options: 
    mean, if we want the mean EoO distance of all pairwise configurations, or max, if we want to have the maximal EoO distance, as 
    in the defintion. Note that the first option is more practical whenever this function is used in training where gradients may 
    be computed, as max produces sparse gradients.

    :param data: (torch.tensor) The full one hot encoded data. Note that if you want to test a classifier for DP, then you
        first have to concatenate the predicted labels in a correct format to the data.
    :param true_labels: (torch.tensor) The true reference labels, required to be used as the reference for calculating the
        TPR and the FPR.
    :param target_feature: (str) The name of the target (label) feature.
    :param protected_feature: (str) The name of the protected feature.
    :param desired_outcome: (str) The label of the desired outcome (positive) with respect to which we calculate the TPR and the FPR.
    :param dataset: (BaseDataset) The instantiated dataset object containing the necessary information for the data.
    :param operation: (str) The operation applied to aggregate the absolute differences in the expected labeling. Available are
        only mean or max. Note that mean is preferable when we have to differentiate through this function.
    :return: (torch.float) The aggregated violation of the equalized odds fairness metric.
    """
    assert operation in ['mean', 'max'], 'Only mean and max operations are available'
    desired_idx = np.argwhere(np.array(dataset.features[target_feature], dtype=str) == str(desired_outcome)).flatten().item()
    reference_target_feature = target_feature + '_ref'
    # add the true labels to the data
    extended_data = torch.cat((data, true_labels), dim=1)
    # extend the index_map
    full_one_hot_index_map = copy(dataset.full_one_hot_index_map)
    full_one_hot_index_map[reference_target_feature] = data.size()[1] + np.arange(true_labels.size()[1])

    target_protected_ref_marginal = query_marginal(
        extended_data, (target_feature, protected_feature, reference_target_feature), full_one_hot_index_map, normalize=True, input_torch=True
    )
    # renormalize to obtain TPR, FPR, TNR, FNR per protected feature
    normalization_constant = target_protected_ref_marginal.sum(0)
    target_protected_ref_marginal = target_protected_ref_marginal / normalization_constant.unsqueeze(0)

    # keep only the PR-s
    tpr_per_protected = target_protected_ref_marginal[desired_idx, :, desired_idx]
    tpr_diffs = tpr_per_protected.view((-1, 1)) - tpr_per_protected.view((1, -1))
    
    op_of_absolute_diffs_tpr = tpr_diffs.abs().mean() if operation == 'mean' else tpr_diffs.abs().max()

    return op_of_absolute_diffs_tpr
    