import numpy as np
from xgboost import XGBClassifier
from denoiser.loss_functions import total_variation_loss, mean_squared_error_loss, jensen_shannon_divergence
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from query import query_marginal
from constraints import ConstraintCompiler, ConstraintEvaluator


def statistics(arr):
    """
    Takes a list or an array and return the mean, std, median, min, and max statistics.

    :param arr: (list or np.ndarray) The data list or array.
    :return: (tuple) Of the mean, std, median, min, and max of the input list or array.
    """
    return np.mean(arr), np.std(arr), np.median(arr), np.min(arr), np.max(arr)


def evaluate_classifier(classifier, Xtrain, ytrain, Xtest, ytest):
    """
    Takes a classifier, training data, and test data, and returns the accuracy, balanced accuracy
    and the f1 score of the classifier. Note that the classifier has to follow the sklearn base
    estimator API.

    :param classifier: (sklearn.base.BaseEstimator) Instantiated classifier that inherits its structure
        fromt the sklearn base estimator abstract class.
    :param Xtrain: (np.ndarray) Training data.
    :param ytrain: (np.ndarray) Training labels.
    :param Xtest: (np.ndarray) Test data.
    :param ytest: (np.ndarray) Test labels.
    :return: (tuple) The accuracy, balanced accuracy, and the f1 score of the classifier in a tuple.
    """
    classifier.fit(Xtrain, ytrain)
    predictions = classifier.predict(Xtest)
    return accuracy_score(ytest, predictions), balanced_accuracy_score(ytest, predictions), f1_score(ytest, predictions, average='micro')


def evaluate_sampled_dataset(synthetic_dataset, workload, true_measured_workload, dataset, max_slice, random_seed=42, classifiers_to_use=['xgb']):
    """
    Takes the synthetic data and the measured marginals on the true data, and returns an evaluation of all the marginal 
    errors on the workload, and the performance statistics of an xgboost.

    :param synthetic_dataset: (torch.tensor) The generated synthetic dataset in full one hot encoding.
    :param workload: (list) The workload as list of tuples.
    :param true_measured_workload: (dict) The true workload measurements.
    :param dataset: (BaseDataset) The instantiated dataset object.
    :param max_slice: (int) Max size for marginal computations.
    :param random_seed: (int) Random seed for reproducibility.
    :param classifiers_to_use: (list) List of names of classifiers to use. 
    :return: (tuple) TV error, L2 error, JS error, XGB accuracy, XGB balanced accuracy, and XGB F1 score statistics, where the last three for each classifier.
    """
    returns = []
    # measure all marginals on it
    all_measured_fake_marginals = {m: query_marginal(synthetic_dataset, m, dataset.full_one_hot_index_map, normalize=True, input_torch=True, max_slice=max_slice) for m in workload}

    returns.append(statistics([total_variation_loss(true_measured_workload[m], all_measured_fake_marginals[m]).item() for m in workload]))
    returns.append(statistics([mean_squared_error_loss(true_measured_workload[m], all_measured_fake_marginals[m]).item() for m in workload]))
    returns.append(statistics([jensen_shannon_divergence(true_measured_workload[m], all_measured_fake_marginals[m]).item() for m in workload]))

    # train classifiers
    Xtest, ytest = ConstraintCompiler.prepare_data(dataset.get_Dtest_full_one_hot(return_torch=True), list(dataset.train_features.keys()), dataset.label, dataset)
    Xtrain_synth, ytrain_synth = ConstraintCompiler.prepare_data(synthetic_dataset.cpu(), list(dataset.train_features.keys()), dataset.label, dataset)

    # avoid encoding error
    ytrain_synth = ConstraintEvaluator.handle_missing_classes_in_training_data(ytrain_synth, dataset.features[dataset.label])
    
    available_classifiers = {
            'logreg': (LogisticRegression, {'random_state': random_seed}),
            'svm': (SVC, {'random_state': random_seed}),
            'rf': (RandomForestClassifier, {'random_state': random_seed}),
            'xgb': (XGBClassifier, {'verbosity': 0, 'random_state': random_seed}),
        }

    for classifier_name in classifiers_to_use:
        classifier = available_classifiers[classifier_name][0](**available_classifiers[classifier_name][1])
        acc, bac, f1 = evaluate_classifier(classifier, Xtrain_synth.cpu().numpy(), ytrain_synth.cpu().numpy().astype(int), Xtest.cpu().numpy(), ytest.cpu().numpy())
        returns.append(statistics([acc]))
        returns.append(statistics([bac]))
        returns.append(statistics([f1]))
    
    return returns
