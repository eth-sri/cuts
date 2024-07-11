import torch


def dl2_geq(val1, val2):
    """
    DL2 loss for greater or equal. Note that at least one of the arguments has to be torch.tensor.
    val1 >= val2.

    :param val1: (float or torch.tensor) Left comparent.
    :param val2: (float or torch.tensor) Right comparent.
    :return: (torch.tensor) Resulting DL2 loss.
    """
    l = torch.nn.functional.relu(val2 - val1)
    return l


def dl2_neq(val1_tensor, val2):
    """
    DL2 loss for not equal.
    val1_tensor != val2.

    :param val1_tensor: (torch.tensor) Left comparent. Has to be a torch.tensor.
    :param val2: (float or torch.tensor) Right comparent.
    :return: (torch.tensor) Resulting DL2 loss.
    """
    l = torch.where(val1_tensor == val2, torch.ones_like(val1_tensor), torch.zeros_like(val1_tensor))
    return l
