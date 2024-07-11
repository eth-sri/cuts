import numpy as np


def create_kfold_index_splits(data_len, k):
    """
    Creat k-fold validation splits over the indices.

    :param data_len: (int) The length of that dataset over which we create the splits.
    :param k: (int) The number of splits to create.
    :return: (list[tuple[np.array]]) A list of k tuples, where the first entry in the tuple
        contains the train indices and the second the validation indices.
    """
    indices = np.arange(data_len)
    np.random.shuffle(indices)
    index_splits = np.array_split(indices, k)

    final_splits = []
    for i in range(k):
        curr_train_split = []
        for j in range(k):
            if j != i:
                curr_train_split.append(index_splits[j])
        final_splits.append((np.concatenate(curr_train_split).flatten(), index_splits[i]))
    return final_splits
