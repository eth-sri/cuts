import torch


def straight_through_softmax(logits: torch.tensor, tau: float = 1.0, dim: int = -1) -> torch.tensor:
    """
    Straight-through estimator for the softmax, based on the straight-through estimator of the gumbel_softmax,
    implemented by pytorch: https://pytorch.org/docs/stable/_modules/torch/nn/functional.html#gumbel_softmax.
    It takes a vector of logits and returns the hard-softmax one-hot vectors, but allows differentiation with respect
    to the soft probabilities. Essentially, the function performs an argmax on the input.

    :param logits: (torch.tensor) The input logits.
    :param tau: (float) Temperature.
    :param dim: (int) Dimension along which the softmax is applied.
    :return: ret (torch.tensor) The hard one-hot vectors from the input logits.
    """
    y_soft = torch.nn.functional.softmax(logits / tau, dim=dim)
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft
    return ret
