import sys
sys.path.append("..")
from abc import ABC, abstractmethod
import torch
from utils import to_categorical, to_numeric, to_ordinal, ordinal_to_categorical, discretize_numerical_features, \
    revert_numerical_features
import numpy as np


class BaseDataset(ABC):

    @abstractmethod
    def __init__(self, name, device, random_state=42):
        self.name = name
        self.device = device
        self.random_state = random_state
        self.split_status = 'train'
        self.standardized = False

        # To be assigned by the concrete dataset
        self.mean, self.std = None, None
        self.mins, self.maxs = None, None
        self.Xtrain, self.ytrain = None, None
        self.Xtest, self.ytest = None, None
        self.Dtrain_full_ordinal, self.Dtest_full_ordinal = None, None
        self.Dtrain_full_one_hot, self.Dtest_full_one_hot = None, None
        self.discretization_ordinal_train, self.discretization_ordinal_test = None, None
        self.discretization_one_hot_train, self.discretization_one_hot_test = None, None
        self.feature_data, self.labels = self.Xtrain, self.ytrain
        self.num_features = None
        self.features, self.train_features = None, None
        self.continuous_features, self.discrete_features = None, None
        self.full_ordinal_specs, self.extended_full_ordinal_specs = None, None
        self.train_full_ordinal_specs, self.train_extended_full_ordinal_specs = None, None
        self.full_one_hot_index_map, self.train_full_one_hot_index_map = None, None
        self.train_bucketing_lower_edges, self.test_bucketing_lower_edges = None, None
        self.index_maps_created = False
        self.histograms_and_continuous_bounds_calculated = False
        self.gmm_parameters_loaded = False
        self.label = ''
        self.single_bit_binary = False

    def __str__(self):
        return self.name + f' Dataset: {self.split_status}'

    def __getitem__(self, item):
        return self.feature_data[item], self.labels[item]

    def __len__(self):
        return self.labels.size()[0]

    def _assign_split(self, split):
        """
        Private method to load data into 'self.features' and 'self.labels' if the object is desired to be used directly.

        :param split: (str) Which split to assign to 'self.features' and 'self.labels'. The available splits are
            ['train', 'test'], meaning we can either assign the training set or the testing set there.
        :return: None
        """
        self.split_status = split
        if split == 'train':
            self.feature_data, self.labels = self.Xtrain, self.ytrain
        elif split == 'test':
            self.feature_data, self.labels = self.Xtest, self.ytest
        else:
            raise ValueError('Unsupported split')

    def train(self):
        self._assign_split('train')

    def test(self):
        self._assign_split('test')
    
    def create_feature_domain_lists(self):
        self.continuous_features = [feature_name for feature_name, feature_domain in self.features.items() if feature_domain is None]
        self.discrete_features = [feature_name for feature_name, feature_domain in self.features.items() if feature_domain is not None]

    def shuffle(self):
        """
        Reshuffles the splits.

        :return: None
        """
        train_shuffle_indices = torch.randperm(self.Xtrain.size()[0]).to(self.device)
        test_shuffle_indices = torch.randperm(self.Xtest.size()[0]).to(self.device)
        self.Xtrain, self.ytrain = self.Xtrain[train_shuffle_indices], self.ytrain[train_shuffle_indices]
        self.Xtest, self.ytest = self.Xtest[test_shuffle_indices], self.ytest[test_shuffle_indices]

    def get_Xtrain(self):
        """
        Returns a detached copy of the training dataset.

        :return: (torch.tensor)
        """
        return self.Xtrain.clone().detach()

    def get_ytrain(self):
        """
        Returns a detached copy of the training labels.

        :return: (torch.tensor)
        """
        return self.ytrain.clone().detach()

    def get_Xtest(self):
        """
        Returns a detached copy of the test dataset.

        :return: (torch.tensor)
        """
        return self.Xtest.clone().detach()

    def get_ytest(self):
        """
        Returns a detached copy of the test labels.

        :return: (torch.tensor)
        """
        return self.ytest.clone().detach()

    def get_Dtrain_full_ordinal(self, buckets=32, return_torch=False):
        """
        Returns the train data in a format where both the continuous and the discrete features are ordinal encoded.

        :param buckets: (int) The number of buckets for discretization of the continuous features.
        :param return_torch: (bool) Toggle to return a torch tensor, else a numpy array is returned.
        :return: (np.ndarray or torch.tensor) The resulting fully ordinal train data.
        """
        if self.Dtrain_full_ordinal is None or self.discretization_ordinal_train != buckets:
            data = self.Xtrain.clone().detach()
            labels = self.ytrain.clone().detach().cpu().numpy().astype(float)
            data = self.encode_ordinal_batch(data, one_hot=True, standardized=self.standardized)
            data = self.discretize_batch(data, buckets=buckets).astype(float)
            data = np.concatenate((data, np.reshape(labels, (-1, 1))), axis=1)
            self.calculate_full_ordinal_specs(buckets)
            self.discretization_ordinal_train = buckets
            self.Dtrain_full_ordinal = data
        else:
            data = self.Dtrain_full_ordinal
        if return_torch:
            return torch.tensor(data, device=self.device)
        else:
            return data

    def get_Dtest_full_ordinal(self, buckets=32, return_torch=False):
        """
        Returns the test data in a format where both the continuous and the discrete features are ordinal encoded.

        :param buckets: (int) The number of buckets for discretization of the continuous features.
        :param return_torch: (bool) Toggle to return a torch tensor, else a numpy array is returned.
        :return: (np.ndarray or torch.tensor) The resulting fully ordinal test data.
        """
        if self.Dtest_full_ordinal is None or self.discretization_ordinal_test != buckets:
            data = self.Xtest.clone().detach()
            labels = self.ytest.clone().detach().cpu().numpy().astype(float)
            data = self.encode_ordinal_batch(data, one_hot=True, standardized=self.standardized)
            data = self.discretize_batch(data, buckets=buckets).astype(float)
            data = np.concatenate((data, np.reshape(labels, (-1, 1))), axis=1)
            self.calculate_full_ordinal_specs(buckets)
            self.discretization_ordinal_test = buckets
            self.Dtest_full_ordinal = data
        else:
            data = self.Dtest_full_ordinal
        if return_torch:
            return torch.tensor(data, device=self.device)
        else:
            return data

    def get_Dtrain_full_one_hot(self, buckets=32, return_torch=False):
        """
        Returns the fully one-hot encoded train data.

        :param buckets: (int) The number of buckets for discretization of the continuous features.
        :param return_torch: (bool) Toggle to return a torch tensor, else a numpy array is returned.
        :return: (np.ndarray or torch.tensor) The resulting fully one-hot encoded train data.
        """
        if self.Dtrain_full_one_hot is None or self.discretization_one_hot_train != buckets:
            full_ordinal_data = self.get_Dtrain_full_ordinal(buckets=buckets, return_torch=False)
            full_one_hot_data = to_numeric(full_ordinal_data, self.extended_full_ordinal_specs)
            self.discretization_one_hot_train = buckets
            self.Dtrain_full_one_hot = full_one_hot_data
            self.train_bucketing_lower_edges = self._calculate_bucketing_lower_edges(buckets)
        else:
            full_one_hot_data = self.Dtrain_full_one_hot
        if return_torch:
            return torch.tensor(full_one_hot_data, device=self.device)
        else:
            return full_one_hot_data

    def get_Dtest_full_one_hot(self, buckets=32, return_torch=False):
        """
        Returns the fully one-hot encoded test data.

        :param buckets: (int) The number of buckets for discretization of the continuous features.
        :param return_torch: (bool) Toggle to return a torch tensor, else a numpy array is returned.
        :return: (np.ndarray or torch.tensor) The resulting fully one-hot encoded test data.
        """
        if self.Dtest_full_one_hot is None or self.discretization_one_hot_test != buckets:
            full_ordinal_data = self.get_Dtest_full_ordinal(buckets=buckets, return_torch=False)
            full_one_hot_data = to_numeric(full_ordinal_data, self.extended_full_ordinal_specs)
            self.discretization_one_hot_test = buckets
            self.Dtest_full_one_hot = full_one_hot_data
            self.test_bucketing_lower_edges = self._calculate_bucketing_lower_edges(buckets)
        else:
            full_one_hot_data = self.Dtest_full_one_hot
        if return_torch:
            return torch.tensor(full_one_hot_data, device=self.device)
        else:
            return full_one_hot_data

    def calculate_full_ordinal_specs(self, buckets=32):
        """
        Updates the full ordinal specs of the dataset with respect to the given discretization.

        :param buckets: (int) The number of buckets for discretization of the continuous features.
        :return: None
        """
        self.full_ordinal_specs = {}
        for feature_name, feature_domain in self.features.items():
            if feature_domain is None:
                self.full_ordinal_specs[feature_name] = buckets
            else:
                self.full_ordinal_specs[feature_name] = len(feature_domain)
        self.extended_full_ordinal_specs = {key: np.arange(val) for key, val in self.full_ordinal_specs.items()}
        self.train_full_ordinal_specs = {key: val for key, val in self.full_ordinal_specs.items() if key != self.label}
        self.train_extended_full_ordinal_specs = {key: val for key, val in self.extended_full_ordinal_specs.items() if key!= self.label}
        pointer = 0
        self.full_one_hot_index_map = {}
        for feature_name, feature_domain in self.extended_full_ordinal_specs.items():
            self.full_one_hot_index_map[feature_name] = pointer + np.arange(len(feature_domain))
            pointer += len(feature_domain)
        self.train_full_one_hot_index_map = {key: val for key, val in self.full_one_hot_index_map.items() if key != self.label}

    def _calculate_mean_std(self):
        """
        Private method to calculate the mean and the standard deviation of the underlying dataset.

        :return: None
        """
        if not self.standardized:  # we do not want to lose the original standardization parameters by overwriting them
            joint_data = torch.cat([self.Xtrain, self.Xtest])
            self.mean = torch.mean(joint_data, dim=0)
            self.std = torch.std(joint_data, dim=0)
            # to avoid divisions by zero and exploding features in constant or nearly constant columns we set standard
            # deviations of zero to one
            zero_stds = torch.nonzero(self.std == 0, as_tuple=False).flatten()
            self.std[zero_stds] = 1.0
    
    def _calculate_mins_maxs(self):
        """
        Private method to calculate the global minimal and maximal values of the continuous features for the dataset.

        :return: None
        """
        Xtrain = self.decode_batch(self.Xtrain.clone(), standardized=self.standardized)
        mins = []
        maxs = []
        for i, (feature_name, feature_domain) in enumerate(self.features.items()):
            if feature_domain is None:
                train_min, train_max = np.min(Xtrain[:, i].astype(float)), np.max(Xtrain[:, i].astype(float))
                mins.append(train_min)
                maxs.append(train_max)
            else:
                continue
        self.mins, self.maxs = mins, maxs

    def standardize(self, batch=None, mode='both'):
        """
        Standardizes the given data (0 mean and 1 variance). It works in three modes: 'batch', 'split', and 'both'. In
        case of 'batch' we standardize a given batch of data by the global statistics of the dataset. In case of 'both'
        we simply standardize the whole underlying dataset, i.e. self.Xtrain and self.Xtest will be standardized. In
        case of 'split' we only standardize the data currently loaded into self.features.

        :param batch:
        :param mode:
        :return: None
        """
        if batch is not None:
            mode = 'batch'
        if mode == 'split':
            self.feature_data = (self.features - self.mean) / self.std
        elif mode == 'both':
            if not self.standardized:
                self.standardized = True
                self.Xtrain, self.Xtest = (self.Xtrain - self.mean) / self.std, (self.Xtest - self.mean) / self.std
                self._assign_split(self.split_status)
        elif mode == 'batch':
            return (batch - self.mean) / self.std
        else:
            raise ValueError('Unsupported mode')

    def de_standardize(self, batch=None, mode='both'):
        """
        Reverts the standardization.

        :param batch: (torch.tensor) The batch to be destandardized, optional.
        :param mode: (str) Choose the mode of standardization, i.e., if all splits ('both'), the current split
            ('split'), or the given batch ('batch') is to be destandardized.
        :return: (None or torch.tensor) If 'batch' mode is selected, we return the destandardized batch, else the
            standardization happens internally.
        """
        if batch is not None:
            mode = 'batch'
        if mode == 'split':
            self.feature_data = self.features * self.std + self.std
        elif mode == 'both':
            if self.standardized:
                self.standardized = False
                self.Xtrain, self.Xtest = self.Xtrain * self.std + self.mean, self.Xtest * self.std + self.mean
                self._assign_split(self.split_status)
        elif mode == 'batch':
            return batch * self.std + self.mean
        else:
            raise ValueError('Unsupported mode')

    def positive_prevalence(self):
        """
        In case of a binary classification task this function calculates the prevalence of the positive label (1). This
        data is useful when assessing the degree of class imbalance.

        :return: (tuple) Prevalence of the positive class in the training set and in the testing set.
        """
        # TODO: for now this method only makes sense for binary classification tasks
        return torch.true_divide(self.ytrain.sum(), self.ytrain.size()[0]), torch.true_divide(self.ytest.sum(), self.ytest.size()[0])

    def decode_batch(self, batch, standardized=True, with_label=False):
        """
        Given a batch of numeric data, this function turns that batch back into the interpretable mixed representation.

        :param batch: (torch.tensor) A batch of data to be decoded according to the features and statistics of the
            underlying dataset.
        :param standardized: (bool) Flag if the batch had been standardized or not.
        :param with_label: (bool) Toggle if the label is included in the batch.
        :return: (np.ndarray) The batch decoded into mixed representation as the dataset is out of the box.
        """
        if standardized:
            batch = self.de_standardize(batch)
        features = self.features if with_label else self.train_features
        return to_categorical(batch.clone().detach().cpu(), features, single_bit_binary=self.single_bit_binary)

    def encode_batch(self, batch, standardize=True, with_label=False):
        """
        Given a batch of mixed type data (np.ndarray on the cpu) we return a numerically encoded batch (torch tensor on
        the dataset device).

        :param batch: (np.ndarray) The mixed type data we wish to convert to numeric.
        :param standardize: (bool) Toggle if the numeric data is to be standardized or not.
        :param with_label: (bool) Toggle if the label is included in the batch.
        :return: (torch.tensor) The numeric encoding of the data as a torch tensor.
        """
        features = self.features if with_label else self.train_features
        batch = torch.tensor(to_numeric(batch, features, label=self.label, single_bit_binary=self.single_bit_binary), device=self.device)
        if standardize:
            batch = self.standardize(batch)
        return batch

    def project_batch(self, batch, standardized=True, with_label=False):
        """
        Given a batch of numeric fuzzy data, this returns its projected encoded counterpart.

        :param batch: (torch.tensor) The fuzzy data to be projected.
        :param standardized: (bool) Mark if the fuzzy data is standardized or not. The data will be returned in the same
            way.
        :param with_label: (bool) Toggle if the label is included in the batch.
        :return: (torch.tensor) The projected data.
        """
        return self.encode_batch(self.decode_batch(batch, standardized=standardized, with_label=with_label), standardize=standardized, with_label=with_label)

    def encode_ordinal_batch(self, batch, one_hot=False, standardized=False, with_label=False):
        """
        Given a batch of mixed type data, or one-hot encoded data (this has to be marked in the extra arguments), this
        function returns the ordinal encoded data.

        :param batch: (np.ndarray or torch.tensor) The data to be encoded as ordinal.
        :param one_hot: (bool) Toggle if the data is one-hot encoded and not mixed-type.
        :param standardized: (bool) Toggle in case of one-hot encoded data if the data is standardized.
        :param with_label: (bool) Toggle if the label is included in the batch.
        :return: (np.ndarray) The ordinal encoded data.
        """
        if one_hot:
            batch = self.decode_batch(batch, standardized=standardized)
        features = self.features if with_label else self.train_features
        return to_ordinal(batch, features).astype(float)

    def decode_ordinal_batch(self, batch, with_label=False):
        """
        Given a batch of ordinal encoded data, we decode it to mixed-type representation.

        :param batch: (np.ndarray) The ordinal encoded data batch.
        :param with_label: (bool) Toggle if the label is included in the batch.
        :return: (np.ndarray) The mixed-type encoding of the same batch.
        """
        features = self.features if with_label else self.train_features
        return ordinal_to_categorical(batch, features)

    def discretize_batch(self, batch, buckets=32, with_label=False):
        """
        Takes the either ordinal or mixed-type encoded data and discretizes the continuous features.

        :param batch: (np.ndarray) The either mixed-type or ordinal encoded data.
        :param buckets: (int) The number of buckets to be used for discretization.
        :param with_label: (bool) Toggle if the label is included in the batch.
        :return: (np.ndarray) The discretized data.
        """
        features = self.features if with_label else self.train_features
        return discretize_numerical_features(batch, features, self.mins, self.maxs, buckets)

    def revert_discretization_batch(self, batch, buckets=32, with_label=False):
        """
        Takes the discretized data and maps the continuous features back to their original space.

        :param batch: (np.ndarray) The either mixed-type or ordinal encoded continuous-discretized batch.
        :param buckets: (int) The number of buckets to used for discretization.
        :param with_label: (bool) Toggle if the label is included in the batch.
        :return: (np.ndarray) The remapped data.
        """
        features = self.features if with_label else self.train_features
        return revert_numerical_features(batch, features, self.mins, self.maxs, buckets)

    def encode_full_ordinal_batch(self, batch, buckets=32, with_label=False, return_torch=False):
        """
        Take a mixed-type batch and return the corresponding full ordinal encoding of it.

        :param batch: (np.ndarray) The mixed type batch to encode in full ordinal.
        :param buckets: (int) The number of buckets for discretitzing the continuous features.
        :param with_label: (bool) Toggle if the data includes the labels.
        :param return_torch: (bool) Toggle if the returned data is desired to be a torch.tensor object.
        :return: (np.ndarray or torch.tensor) The resulting ordinal encoded data.
        """
        full_ordinal_batch = self.encode_ordinal_batch(batch, with_label=with_label)
        full_ordinal_batch = self.discretize_batch(full_ordinal_batch, buckets, with_label=with_label)
        self.calculate_full_ordinal_specs(buckets)
        if return_torch:
            return torch.tensor(full_ordinal_batch, device=self.device)
        else:
            return full_ordinal_batch.astype(float)

    def encode_full_one_hot_batch(self, batch, buckets=32, with_label=False, already_ordinal=False, return_torch=False):
        """
        Take a batch and return the corresponding full one-hot encoding of it.

        :param batch: (np.ndarray) The mixed type batch to encode in full one-hot.
        :param buckets: (int) The number of buckets for discretitzing the continuous features.
        :param with_label: (bool) Toggle if the data includes the labels.
        :param already_ordinal: (bool) Toggle if the batch is already ordinal encoded. Pay attention to the buckets.
        :param return_torch: (bool) Toggle if the returned data is desired to be a torch.tensor object.
        :return: (np.ndarray or torch.tensor) The resulting one-hot encoded data.
        """
        if not already_ordinal:
            full_ordinal_batch = self.encode_full_ordinal_batch(batch, buckets, with_label, return_torch=False)
        else:
            # make sure that the specs are calculated
            # self.calculate_full_ordinal_specs(buckets) -- TODO: this is for some reason not working although it should
            full_ordinal_batch = batch
        specs = self.extended_full_ordinal_specs if with_label else self.train_extended_full_ordinal_specs
        full_one_hot_batch = to_numeric(full_ordinal_batch, specs)
        if return_torch:
            return torch.tensor(full_one_hot_batch, device=self.device)
        else:
            return full_one_hot_batch

    def decode_full_ordinal_batch(self, batch, buckets=32, with_label=False, input_torch=False):
        """
        Takes a fully ordinal batch and returns the mixed-type encoded pair of it.

        :param batch: (np.ndarrad or torch.tensor) The fully ordinal batch to be decoded.
        :param buckets: (int) The number of buckets to used for discretization.
        :param with_label: (bool) Toggle if the label is included in the batch.
        :param input_torch: (bool) Toggle if the input is a torch.tensor.
        :return: (np.ndarray) The corresponding mixed-type data.
        """
        if input_torch:
            batch = batch.detach().cpu().numpy()
        return self.decode_ordinal_batch(self.revert_discretization_batch(batch, buckets, with_label), with_label)

    def decode_full_one_hot_batch(self, batch, buckets=32, with_label=False, input_torch=False):
        """
        Takes a fully one-hot batch and returns the mixed-type encoded pair of it.

        :param batch: (np.ndarrad or torch.tensor) The fully one-hot batch to be decoded.
        :param buckets: (int) The number of buckets to used for discretization.
        :param with_label: (bool) Toggle if the label is included in the batch.
        :param input_torch: (bool) Toggle if the input is a torch.tensor.
        :return: (np.ndarray) The corresponding mixed-type data.
        """
        if input_torch:
            batch = batch.detach().cpu().numpy()
        specs = self.extended_full_ordinal_specs if with_label else self.train_extended_full_ordinal_specs
        full_ordinal_batch = to_categorical(batch, features=specs).astype(float)
        return self.decode_full_ordinal_batch(full_ordinal_batch, buckets=buckets, with_label=with_label)

    def _create_index_maps(self):
        """
        A private method that creates easy access indexing tools for other methods.

        :return: None
        """
        # check if the feature map has already been assigned
        assert self.features is not None, 'Instantiate a dataset with a feature map'

        # register the type of the feature and the positions of all numerical features corresponding to this feature
        pointer = 0
        self.feature_index_map = {}
        for val, key in zip(self.features.values(), self.features.keys()):
            if val is None or (len(val) == 2 and self.single_bit_binary):
                index_list = [pointer]
                pointer += 1
            else:
                index_list = [pointer + i for i in range(len(val))]
                pointer += len(val)
            im = 'cont' if val is None else 'cat'
            self.feature_index_map[key] = (im, index_list) if key != self.label else (im, index_list[0])  # binary labels

        # for ease of use, create the one just for the X part of the data
        self.train_feature_index_map = {key: self.feature_index_map[key] for key in self.train_features.keys()}

        # for ease of use make the non-numerically encoded feature positions also accessible by type
        index_map = np.array(['cont' if val is None else 'cat' for val in self.features.values()])
        self.cont_indices = np.argwhere(index_map == 'cont').flatten()
        self.cat_indices = np.argwhere(index_map == 'cat').flatten()
        # this makes sense only for classification tasks
        self.train_cont_indices = self.cont_indices
        self.train_cat_indices = self.cat_indices[:-1]
        self.index_maps_created = True

    def _calculate_categorical_feature_distributions_and_continuous_bounds(self):
        """
        A private method to calculate the feature distributions and feature bounds that are needed to understand the
        statistical properties of the dataset.

        :return: None
        """
        # if we do not have the index maps yet then we should create that
        if not self.index_maps_created:
            self._create_index_maps()

        # copy the feature tensors and concatenate them
        X = torch.cat([self.get_Xtrain(), self.get_Xtest()], dim=0)

        # check if the dataset was standardized, if yes then destandardize X
        if self.standardized:
            X = self.de_standardize(X)

        # now run through X and create the necessary items
        X = X.detach().clone().cpu().numpy()
        n_samples = X.shape[0]
        self.categorical_histograms = {}
        self.cont_histograms = {}
        self.continuous_bounds = {}
        self.standardized_continuous_bounds = {}

        for key, (feature_type, index_map) in self.train_feature_index_map.items():
            if feature_type == 'cont':
                # calculate the bounds
                lb = min(X[:, index_map[0]])
                ub = max(X[:, index_map[0]])
                self.continuous_bounds[key] = (lb, ub)
                self.standardized_continuous_bounds[key] = ((lb - self.mean[index_map].item()) / self.std[index_map].item(),
                                                            (ub - self.mean[index_map].item()) / self.std[index_map].item())
                # calculate histograms
                value_range = np.arange(lb, ub+1)
                hist, _ = np.histogram(X[:, index_map[0]], bins=min(100, len(value_range)))
                self.cont_histograms[key] = hist / n_samples
            elif feature_type == 'cat':
                # calculate the histograms
                hist = np.sum(X[:, index_map], axis=0) / n_samples
                # extend the histogram to two entries for binary features (Bernoulli dist)
                if len(hist) == 1:
                    hist = np.array([1-hist[0], hist[0]])
                self.categorical_histograms[key] = hist
            else:
                raise ValueError('Invalid feature index map')
        self.histograms_and_continuous_bounds_calculated = True

    def create_tolerance_map(self, tol=0.319):
        """
        Given a tolerance value for multiplying the standard deviation, this method calculates a tolerance map that is
        required for the error calculation between a guessed/reconstructed batch and a true batch of data.

        :param tol: (float) Tolerance value. The tolerance interval for each continuous feature will be calculated as:
            [true - tol, true + tol].
        :return: (list) The tolerance map required for the error calculation.
        """
        x_std = self.std.clone().detach().cpu().numpy()
        cont_indices = [idxs[0] for nature, idxs in self.train_feature_index_map.values() if nature == 'cont']
        numeric_stds = x_std[cont_indices]
        tolerance_map = []
        pointer = 0

        for value in self.train_features.values():
            to_append = tol * numeric_stds[pointer] if value is None else 'cat'
            pointer += 1 if value is None else 0
            tolerance_map.append(to_append)

        return tolerance_map

    def _calculate_bucketing_lower_edges(self, buckets=32):
        """
        Takes a bucketing and returns the corresponding lower edges of the bucketing over the whole domain.

        :param buckets: (int) Number of buckets in the bucketing.
        :return: (dict) The lower bucket edges in a dictionary.
        """
        edges = ((np.arange(buckets).reshape((-1, 1)))*(np.array(self.maxs)-np.array(self.mins))/buckets + np.array(self.mins)).T
        edges_dict = {}
        for lb_line, feature in zip(edges, self.continuous_features):
            edges_dict[feature] = lb_line
        
        return edges_dict
