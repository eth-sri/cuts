import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
import os

sys.path.append("..")
from utils import to_numeric
from sklearn.model_selection import train_test_split


class German(BaseDataset):

    def __init__(self, name='German', train_test_ratio=0.2, single_bit_binary=False, device='cpu', random_state=42, split_from_file=True):
        super(German, self).__init__(name=name, device=device, random_state=random_state)

        self.train_test_ratio = train_test_ratio

        self.features = German.get_features()

        self.single_bit_binary = single_bit_binary
        self.label = 'class'

        self.train_features = {key: self.features[key] for key in self.features.keys() if key != self.label}

        if os.path.isfile(f'tabular_datasets/German/presplit_xtrain_{train_test_ratio}_{random_state}.npy') and split_from_file:
            Xtrain = np.load(f'tabular_datasets/German/presplit_xtrain_{train_test_ratio}_{random_state}.npy')
            ytrain = np.load(f'tabular_datasets/German/presplit_ytrain_{train_test_ratio}_{random_state}.npy')
            Xtest = np.load(f'tabular_datasets/German/presplit_xtest_{train_test_ratio}_{random_state}.npy')
            ytest = np.load(f'tabular_datasets/German/presplit_ytest_{train_test_ratio}_{random_state}.npy')
            self.num_features = Xtrain.shape[1]
        
        else:
            # load the data
            data_df = pd.read_csv('tabular_datasets/German/german.data', delimiter=' ', names=list(self.features.keys()), engine='python')

            # convert to numeric
            data = data_df.to_numpy()
            data_num = (to_numeric(data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)).astype(np.float32)

            # split labels and features
            X, y = data_num[:, :-1], data_num[:, -1]
            self.num_features = X.shape[1]

            # create a train and test split and shuffle
            Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=self.train_test_ratio,
                                                            random_state=self.random_state, shuffle=True)

        # convert to torch
        self.Xtrain, self.Xtest = torch.tensor(Xtrain).to(self.device), torch.tensor(Xtest).to(self.device)
        self.ytrain, self.ytest = torch.tensor(ytrain, dtype=torch.long).to(self.device), torch.tensor(ytest, dtype=torch.long).to(self.device)

        # set to train mode as base
        self.train()

        # calculate the standardization statistics
        self._calculate_mean_std()

        # calculate the mins and the maxs
        self._calculate_mins_maxs()

        # calculate the histograms and feature bounds
        self._calculate_categorical_feature_distributions_and_continuous_bounds()

        # fill the feature domain lists
        self.create_feature_domain_lists()

    def repeat_split(self, split_ratio=None, random_state=None):
        """
        As the dataset does not come with a standard train-test split, we assign this split manually during the
        initialization. To allow for independent experiments without much of a hassle, we allow through this method for
        a reassignment of the split.

        :param split_ratio: (float) The desired ratio of test_data/all_data.
        :param random_state: (int) The random state according to which we do the assignment,
        :return: None
        """
        if random_state is None:
            random_state = self.random_state
        if split_ratio is None:
            split_ratio = self.train_test_ratio
        X = torch.cat([self.Xtrain, self.Xtest], dim=0).detach().cpu().numpy()
        y = torch.cat([self.ytrain, self.ytest], dim=0).detach().cpu().numpy()
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=split_ratio, random_state=random_state,
                                                        shuffle=True)
        # convert to torch
        self.Xtrain, self.Xtest = torch.tensor(Xtrain).to(self.device), torch.tensor(Xtest).to(self.device)
        self.ytrain, self.ytest = torch.tensor(ytrain, dtype=torch.long).to(self.device), torch.tensor(ytest, dtype=torch.long).to(self.device)
        # update the split status as well
        self._assign_split(self.split_status)

    @staticmethod
    def get_features():
        """
        Static method such that we can access the features of the dataset without having to instantiate it.

        :return: (dict) The features.
        """
        features = {
            'A1': ['A1' + str(i) for i in range(1, 5)],  # status of existing checking account
            'A2': None,  # duration
            'A3': ['A3' + str(i) for i in range(0, 5)],  # credit history
            'A4': ['A4' + str(i) for i in range(0, 11)],  # purpose
            'A5': None,  # credit amount
            'A6': ['A6' + str(i) for i in range(1, 6)],  # savings account/bonds
            'A7': ['A7' + str(i) for i in range(1, 6)],  # present employment since
            'A8': None,  # installment rate in percentage of dispsable income
            'A9': ['A9' + str(i) for i in range(1, 6)],  # personal status and sex
            'A10': ['A10' + str(i) for i in range(1, 4)],  # other debtors / guarantors
            'A11': None,  # present residence since
            'A12': ['A12' + str(i) for i in range(1, 5)],  # property
            'A13': None,  # age
            'A14': ['A14' + str(i) for i in range(1, 4)],  # other installment plans
            'A15': ['A15' + str(i) for i in range(1, 4)],  # housing
            'A16': None,  # number of existing credits at this bank
            'A17': ['A17' + str(i) for i in range(1, 5)],  # job
            'A18': None,  # number of people being liable to provide maintanance for
            'A19': ['A19' + str(i) for i in range(1, 3)],  # telephone
            'A20': ['A20' + str(i) for i in range(1, 3)],  # foreign worker
            'class': [1, 2]  # credit risk good or bad
        }
       
        return features