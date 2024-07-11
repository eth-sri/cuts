import torch
import numpy as np
import pandas as pd
from .base_dataset import BaseDataset
import sys
import os

sys.path.append("..")
from utils import to_numeric, to_categorical
from sklearn.model_selection import train_test_split


class HealthHeritage(BaseDataset):

    def __init__(self, name='HealthHeritage', train_test_ratio=0.2, binary_age=False, single_bit_binary=False, device='cpu',
                 split_from_file=True, random_state=42):
        super(HealthHeritage, self).__init__(name=name, device=device, random_state=random_state)

        self.train_test_ratio = train_test_ratio

        self.single_bit_binary = single_bit_binary
        self.label = 'max_CharlsonIndex'

        self.binary_age = binary_age

        if os.path.isfile(f'tabular_datasets/Health_Heritage/presplit_xtrain_{train_test_ratio}_{random_state}_{binary_age}.npy') and split_from_file:
            Xtrain = np.load(f'tabular_datasets/Health_Heritage/presplit_xtrain_{train_test_ratio}_{random_state}_{binary_age}.npy')
            ytrain = np.load(f'tabular_datasets/Health_Heritage/presplit_ytrain_{train_test_ratio}_{random_state}_{binary_age}.npy')
            Xtest = np.load(f'tabular_datasets/Health_Heritage/presplit_xtest_{train_test_ratio}_{random_state}_{binary_age}.npy')
            ytest = np.load(f'tabular_datasets/Health_Heritage/presplit_ytest_{train_test_ratio}_{random_state}_{binary_age}.npy')

            self.features = HealthHeritage.get_features(binary_age=binary_age)
            self.features['max_CharlsonIndex'] = ['=0', '>0']
            self.train_features = {key: self.features[key] for key in self.features.keys() if key != self.label}
            # reorder the features directory to put the labels at the back
            del self.features['max_CharlsonIndex']
            self.features['max_CharlsonIndex'] = ['=0', '>0']
            self.num_features = Xtrain.shape[1]

        else: 

            # preprocess the data according to https://github.com/truongkhanhduy95/Heritage-Health-Prize
            df_claims = self.preprocess_claims(pd.read_csv('tabular_datasets/Health_Heritage/Claims.csv', sep=','),
                                        ['PrimaryConditionGroup', 'Specialty', 'ProcedureGroup', 'PlaceSvc'])
            df_drugs = self.preprocess_drugs(pd.read_csv('tabular_datasets/Health_Heritage/DrugCount.csv', sep=','))
            df_labs = self.preprocess_labs(pd.read_csv('tabular_datasets/Health_Heritage/LabCount.csv', sep=','))
            df_members = self.preprocess_members(pd.read_csv('tabular_datasets/Health_Heritage/Members.csv', sep=','))

            df_labs_drugs = pd.merge(df_labs, df_drugs, on=['MemberID', 'Year'], how='outer')
            df_labs_drugs_claims = pd.merge(df_labs_drugs, df_claims, on=['MemberID', 'Year'], how='outer')
            df_health = pd.merge(df_labs_drugs_claims, df_members, on=['MemberID'], how='outer')

            df_health.drop(['Year', 'MemberID'], axis=1, inplace=True)
            df_health.fillna(0, inplace=True)

            self.features = HealthHeritage.get_features(binary_age=False)
            health_np = df_health.to_numpy()
            health_np_mixed = to_categorical(health_np, self.features)
            
            if self.binary_age:
                protected = ['60-69', '70-79', '80+']
                df_health = pd.DataFrame(data=health_np_mixed, columns=list(self.features.keys()))
                df_health['AgeAtFirstClaim'] = df_health['AgeAtFirstClaim'].apply(lambda x: '>=60' if x in protected else '<60')
                self.features = HealthHeritage.get_features(binary_age=True)
                health_np_mixed = df_health.to_numpy()

            # create the labels
            for i, feature in enumerate(self.features.keys()):
                if feature == 'max_CharlsonIndex':
                    for j in range(len(health_np_mixed[:, i])):
                        health_np_mixed[j, i] = '>0' if (float(health_np_mixed[j, i]) > 0) else '=0'

            self.features['max_CharlsonIndex'] = ['=0', '>0']
            self.train_features = {key: self.features[key] for key in self.features.keys() if key != self.label}

            # convert back to now already the correct numeric representation
            health_np = to_numeric(health_np_mixed, self.features, label=self.label).astype(np.float32)

            # split the labels and the features
            non_label_indices = [i for i in range(health_np.shape[1]) if i != 8]
            X, y = health_np[:, non_label_indices], health_np[:, 8]
            self.num_features = X.shape[1]

            # reorder the features directory to put the labels at the back
            del self.features['max_CharlsonIndex']
            self.features['max_CharlsonIndex'] = ['=0', '>0']

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
    def get_features(binary_age=False):
        """
        Static method such that we can access the features of the dataset without having to instantiate it.

        :param binary_age: (bool) Toggle if the AgeAtFirstClaim feature is binarized to over and under 60.
        :return: (dict) The features.
        """
        features = {
            'LabCount_total': None,
            'LabCount_months': None,
            'DrugCount_total': None,
            'DrugCount_months': None,
            'no_Claims': None,
            'no_Providers': None,
            'no_Vendors': None,
            'no_PCPs': None,
            'max_CharlsonIndex': None,
            'PayDelay_total': None,
            'PayDelay_max': None,
            'PayDelay_min': None,
            'PrimaryConditionGroup': ['AMI', 'APPCHOL', 'ARTHSPIN', 'CANCRA', 'CANCRB', 'CANCRM', 'CATAST', 'CHF',
                                      'COPD', 'FLaELEC', 'FXDISLC', 'GIBLEED',
                                      'GIOBSENT', 'GYNEC1', 'GYNECA', 'HEART2', 'HEART4', 'HEMTOL', 'HIPFX', 'INFEC4',
                                      'LIVERDZ', 'METAB1', 'METAB3',
                                      'MISCHRT', 'MISCL1', 'MISCL5', 'MSC2a3', 'NEUMENT', 'ODaBNCA', 'PERINTL',
                                      'PERVALV', 'PNCRDZ', 'PNEUM', 'PRGNCY',
                                      'PrimaryConditionGroup_?', 'RENAL1', 'RENAL2', 'RENAL3', 'RESPR4', 'ROAMI',
                                      'SEIZURE', 'SEPSIS', 'SKNAUT', 'STROKE',
                                      'TRAUMA', 'UTI'],
            'Specialty': ['Anesthesiology', 'Diagnostic Imaging', 'Emergency', 'General Practice', 'Internal',
                          'Laboratory', 'Obstetrics and Gynecology',
                          'Other', 'Pathology', 'Pediatrics', 'Rehabilitation', 'Specialty_?', 'Surgery'],
            'ProcedureGroup': ['ANES', 'EM', 'MED', 'PL', 'ProcedureGroup_?', 'RAD', 'SAS', 'SCS', 'SDS', 'SEOA', 'SGS',
                               'SIS', 'SMCD', 'SMS', 'SNS',
                               'SO', 'SRS', 'SUS'],
            'PlaceSvc': ['Ambulance', 'Home', 'Independent Lab', 'Inpatient Hospital', 'Office', 'Other',
                         'Outpatient Hospital', 'PlaceSvc_?', 'Urgent Care'],
            'AgeAtFirstClaim': ['0-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70-79', '80+', '?'],
            'Sex': ['?', 'F', 'M']
        }

        if binary_age:
            features['AgeAtFirstClaim'] = ['>=60', '<60']
       
        return features

    @staticmethod
    def preprocess_claims(df_claims, cat_names):
        df_claims.loc[df_claims['PayDelay'] == '162+', 'PayDelay'] = 162
        df_claims['PayDelay'] = df_claims['PayDelay'].astype(int)

        df_claims.loc[df_claims['DSFS'] == '0- 1 month', 'DSFS'] = 1
        df_claims.loc[df_claims['DSFS'] == '1- 2 months', 'DSFS'] = 2
        df_claims.loc[df_claims['DSFS'] == '2- 3 months', 'DSFS'] = 3
        df_claims.loc[df_claims['DSFS'] == '3- 4 months', 'DSFS'] = 4
        df_claims.loc[df_claims['DSFS'] == '4- 5 months', 'DSFS'] = 5
        df_claims.loc[df_claims['DSFS'] == '5- 6 months', 'DSFS'] = 6
        df_claims.loc[df_claims['DSFS'] == '6- 7 months', 'DSFS'] = 7
        df_claims.loc[df_claims['DSFS'] == '7- 8 months', 'DSFS'] = 8
        df_claims.loc[df_claims['DSFS'] == '8- 9 months', 'DSFS'] = 9
        df_claims.loc[df_claims['DSFS'] == '9-10 months', 'DSFS'] = 10
        df_claims.loc[df_claims['DSFS'] == '10-11 months', 'DSFS'] = 11
        df_claims.loc[df_claims['DSFS'] == '11-12 months', 'DSFS'] = 12

        df_claims.loc[df_claims['CharlsonIndex'] == '0', 'CharlsonIndex'] = 0
        df_claims.loc[df_claims['CharlsonIndex'] == '1-2', 'CharlsonIndex'] = 1
        df_claims.loc[df_claims['CharlsonIndex'] == '3-4', 'CharlsonIndex'] = 2
        df_claims.loc[df_claims['CharlsonIndex'] == '5+', 'CharlsonIndex'] = 3

        df_claims.loc[df_claims['LengthOfStay'] == '1 day', 'LengthOfStay'] = 1
        df_claims.loc[df_claims['LengthOfStay'] == '2 days', 'LengthOfStay'] = 2
        df_claims.loc[df_claims['LengthOfStay'] == '3 days', 'LengthOfStay'] = 3
        df_claims.loc[df_claims['LengthOfStay'] == '4 days', 'LengthOfStay'] = 4
        df_claims.loc[df_claims['LengthOfStay'] == '5 days', 'LengthOfStay'] = 5
        df_claims.loc[df_claims['LengthOfStay'] == '6 days', 'LengthOfStay'] = 6
        df_claims.loc[df_claims['LengthOfStay'] == '1- 2 weeks', 'LengthOfStay'] = 11
        df_claims.loc[df_claims['LengthOfStay'] == '2- 4 weeks', 'LengthOfStay'] = 21
        df_claims.loc[df_claims['LengthOfStay'] == '4- 8 weeks', 'LengthOfStay'] = 42
        df_claims.loc[df_claims['LengthOfStay'] == '26+ weeks', 'LengthOfStay'] = 180
        df_claims['LengthOfStay'].fillna(0, inplace=True)
        df_claims['LengthOfStay'] = df_claims['LengthOfStay'].astype(int)

        for cat_name in cat_names:
            df_claims[cat_name].fillna(f'{cat_name}_?', inplace=True)
        df_claims = pd.get_dummies(df_claims, columns=cat_names, prefix_sep='=')

        oh = [col for col in df_claims if '=' in col]

        agg = {
            'ProviderID': ['count', 'nunique'],
            'Vendor': 'nunique',
            'PCP': 'nunique',
            'CharlsonIndex': 'max',
            # 'PlaceSvc': 'nunique',
            # 'Specialty': 'nunique',
            # 'PrimaryConditionGroup': 'nunique',
            # 'ProcedureGroup': 'nunique',
            'PayDelay': ['sum', 'max', 'min']
        }
        for col in oh:
            agg[col] = 'sum'

        df_group = df_claims.groupby(['Year', 'MemberID'])
        df_claims = df_group.agg(agg).reset_index()
        df_claims.columns = [
                                'Year', 'MemberID', 'no_Claims', 'no_Providers', 'no_Vendors', 'no_PCPs',
                                'max_CharlsonIndex', 'PayDelay_total', 'PayDelay_max', 'PayDelay_min'
                            ] + oh

        return df_claims

    @staticmethod
    def preprocess_drugs(df_drugs):
        df_drugs.drop(columns=['DSFS'], inplace=True)
        # df_drugs['DSFS'] = df_drugs['DSFS'].apply(lambda x: int(x.split('-')[0])+1)
        df_drugs['DrugCount'] = df_drugs['DrugCount'].apply(lambda x: int(x.replace('+', '')))
        df_drugs = df_drugs.groupby(['Year', 'MemberID']).agg({'DrugCount': ['sum', 'count']}).reset_index()
        df_drugs.columns = ['Year', 'MemberID', 'DrugCount_total', 'DrugCount_months']
        # print('df_drugs.shape = ', df_drugs.shape)
        return df_drugs

    @staticmethod
    def preprocess_labs(df_labs):
        df_labs.drop(columns=['DSFS'], inplace=True)
        # df_labs['DSFS'] = df_labs['DSFS'].apply(lambda x: int(x.split('-')[0])+1)
        df_labs['LabCount'] = df_labs['LabCount'].apply(lambda x: int(x.replace('+', '')))
        df_labs = df_labs.groupby(['Year', 'MemberID']).agg({'LabCount': ['sum', 'count']}).reset_index()
        df_labs.columns = ['Year', 'MemberID', 'LabCount_total', 'LabCount_months']
        # print('df_labs.shape = ', df_labs.shape)
        return df_labs

    @staticmethod
    def preprocess_members(df_members):
        df_members['AgeAtFirstClaim'].fillna('?', inplace=True)
        df_members['Sex'].fillna('?', inplace=True)
        df_members = pd.get_dummies(
            df_members, columns=['AgeAtFirstClaim', 'Sex'], prefix_sep='='
        )
        # print('df_members.shape = ', df_members.shape)
        return df_members
