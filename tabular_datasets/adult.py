import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
sys.path.append("..")
from utils import to_numeric
import pickle


class ADULT(BaseDataset):

    def __init__(self, name='ADULT', drop_education_num=True, single_bit_binary=False, device='cpu', random_state=42):
        super(ADULT, self).__init__(name=name, device=device, random_state=random_state)

        self.features = ADULT.get_features(drop_education_num=False)

        self.single_bit_binary = single_bit_binary
        self.label = 'salary'

        self.train_features = {key: self.features[key] for key in self.features.keys() if key != self.label}

        train_data_df = pd.read_csv('tabular_datasets/ADULT/adult.data', delimiter=', ', names=list(self.features.keys()), engine='python')
        test_data_df = pd.read_csv('tabular_datasets/ADULT/adult.test', delimiter=', ', names=list(self.features.keys()), skiprows=1, engine='python')

        if drop_education_num:
            train_data_df = train_data_df.drop(['education-num'], axis=1)
            test_data_df = test_data_df.drop(['education-num'], axis=1)
            del self.features['education-num']
            del self.train_features['education-num']
        
        train_data = train_data_df.to_numpy()
        test_data = test_data_df.to_numpy()

        # drop missing values
        # note that the category never worked always comes with a missing value for the occupation field, hence this
        # step effectively removes the never worked category from the dataset
        train_rows_to_keep = [not ('?' in row) for row in train_data]
        test_rows_to_keep = [not ('?' in row) for row in test_data]
        train_data = train_data[train_rows_to_keep]
        test_data = test_data[test_rows_to_keep]

        # remove the annoying dot from the test labels
        for row in test_data:
            row[-1] = row[-1][:-1]

        # convert to numeric features
        train_data_num = to_numeric(train_data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)
        test_data_num = to_numeric(test_data, self.features, label=self.label, single_bit_binary=self.single_bit_binary)

        # split features and labels
        Xtrain, Xtest = train_data_num[:, :-1].astype(np.float32), test_data_num[:, :-1].astype(np.float32)
        ytrain, ytest = train_data_num[:, -1].astype(np.float32), test_data_num[:, -1].astype(np.float32)
        self.num_features = Xtrain.shape[1]

        # transfer to torch
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

    @staticmethod
    def get_features(drop_education_num=True):
        """
        Static method such that we can access the features of the dataset without having to instantiate it.

        :param drop_education_num: (bool) Toggle to drop the education-num feature.
        :return: (dict) The features.
        """
        features = {
            'age': None,
            'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov',
                          'Without-pay', 'Never-worked'],
            'fnlwgt': None,
            'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc',
                          '9th', '7th-8th', '12th', 'Masters',
                          '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
            'education-num': None,
            'marital-status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed',
                               'Married-spouse-absent', 'Married-AF-spouse'],
            'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial',
                           'Prof-specialty', 'Handlers-cleaners',
                           'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving',
                           'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
            'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
            'race': ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black'],
            'sex': ['Female', 'Male'],
            'capital-gain': None,
            'capital-loss': None,
            'hours-per-week': None,
            'native-country': ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada', 'Germany',
                               'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
                               'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy', 'Poland',
                               'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland',
                               'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti', 'Columbia',
                               'Hungary', 'Guatemala', 'Nicaragua', 'Scotland',
                               'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru', 'Hong',
                               'Holand-Netherlands'],
            'salary': ['>50K', '<=50K']
        }

        if drop_education_num:
            del features['education-num']

        return features
