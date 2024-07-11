import torch
from utils import straight_through_softmax, demographic_parity_distance, equalized_odds_distance, equality_of_opportunity_distance
from classification_models import MetaMonkey
import numpy as np
from collections import OrderedDict


def train_model_differentiable_monkey(Xtr, ytr, monkey_model, criterion, lr=0.01, batch_size=512, num_epochs=10):
    """
    This function takes training data and a monkey patched model and trains it allowing for backpropagation 
    through the training of the model.

    :param Xtr: (torch.tensor) Training data.
    :param ytr: (torch.tensor) Training labels.
    :param monkey_model: (nn.Module) Monkey patched model that is to be trained.
    :param lr: (float) Learining rate for gradient descent.
    :param criterion: (torch.nn Loss function) The instantiated loss function for training.
    :param batch_size: (int) Batch size for training.
    :param num_epochs: (int) Number of epochs of the training.
    :return: None
    """
    num_batches = np.ceil(len(Xtr)/batch_size).astype(int)
    for it in range(num_epochs):
        for bn in range(num_batches):
            current_batch_x, current_batch_y = Xtr[bn*batch_size:min((bn+1)*batch_size, len(Xtr))], ytr[bn*batch_size:min((bn+1)*batch_size, len(ytr))]
            output = monkey_model(current_batch_x, monkey_model.parameters)
            loss = criterion(output, current_batch_y)
            grad = torch.autograd.grad(loss, monkey_model.parameters.values(), retain_graph=True, create_graph=True, only_inputs=True, allow_unused=True)
            monkey_model.parameters = OrderedDict((name, param - lr * param_grad) for ((name, param), param_grad) in zip(monkey_model.parameters.items(), grad))
            # print(1- (straight_through_softmax(output) - current_batch_y).abs().item()/2.0)


def downstream_prediction(X_train, y_train, X_eval, y_eval, model, lr=0.01, batch_size=512, num_epochs=10):
    """
    This function takes a differentiabble model, a training dataset, and an evaluation dataset; and returns the loss
    of the trained model on the evaluation data. As it can be differentiated through, it can be used to optimize the
    evaluation loss, by adjusting the training data fed to the model. Pay attention never to use the real test data as
    the evaluation dataset.

    :param X_train: (torch.tensor) Training features one-hot encoded.
    :param y_train: (torch.tensor of torch.long) Training labels, ordinal encoded.
    :param X_eval: (torch.tensor) Evaluation features one-hot encoded.
    :param y_eval: (torch.tensor of torch.long) Evaluation labels, ordinal encoded.
    :param model: (torch.nn.Module) Instantiated differentiable torch classification model. Make sure that the model is 
        supported by monkey patcing.
    :param lr: (float) Learning rate for training the model with gradient descent.
    :param batch_size: (int) Batch size for training.
    :param num_epochs: (int) Number of epochs for training.
    :return: (torch.float) The loss of the trained model on the evaluation data.
    """
    # monkey patch the model to be able to do backprop through it
    monkey_model = MetaMonkey(model)
    
    # train the model
    criterion = torch.nn.CrossEntropyLoss() 
    train_model_differentiable_monkey(
        Xtr=X_train, 
        ytr=y_train, 
        monkey_model=monkey_model, 
        criterion=criterion, 
        lr=lr, 
        batch_size=batch_size, 
        num_epochs=num_epochs
    )
    
    # calculate the loss on the true data
    loss = criterion(monkey_model(X_eval, monkey_model.parameters), y_eval)
    
    return loss


def downstream_demographic_parity(X_train, y_train, X_eval, y_eval, model, dataset, protected_feature, 
                                  target_feature=None, lr=0.01, batch_size=512, num_epochs=10):
    """
    This function takes a differentiable model, a training dataset, and an evaluation dataset, and computes the
    demographic parity distance of the trained model on the evaluation dataset with respect to a given feature
    in a differentiable manner.

    :param X_train: (torch.tensor) Training features one-hot encoded.
    :param y_train: (torch.tensor of torch.long) Training labels, ordinal encoded.
    :param X_eval: (torch.tensor) Evaluation features one-hot encoded.
    :param y_eval: (torch.tensor of torch.long) Evaluation labels, ordinal encoded. Note that in this function they
        are not used, and could be set to None.
    :param model: (torch.nn.Module) Instantiated differentiable torch classification model. Make sure that the model is 
        supported by monkey patcing.
    :param dataset: (BaseDataset) The instantiated dataset object containing the necessary information for the data.
    :param protected_feature: (str) The name of the protected feature.
    :param target_feature: (str) The name of the target (label) feature.
    :param lr: (float) Learning rate for training the model with gradient descent.
    :param batch_size: (int) Batch size for training.
    :param num_epochs: (int) Number of epochs for training.
    :return: (torch.tensor) The demographic parity distance of the trained model on the evaluationd data.
    """
    if target_feature is None:
        target_feature = dataset.label
    
    # monkey patch the model to be able to do backrpop through it
    monkey_model = MetaMonkey(model)
    
    # train the model
    criterion = torch.nn.CrossEntropyLoss() 
    train_model_differentiable_monkey(
        Xtr=X_train, 
        ytr=y_train, 
        monkey_model=monkey_model, 
        criterion=criterion, 
        lr=lr, 
        batch_size=batch_size, 
        num_epochs=num_epochs
    )
    
    # make predictions on the evaluation data
    predictions = monkey_model(X_eval, monkey_model.parameters)
    predictions_hardened = straight_through_softmax(predictions)
    joint_evaluation_data_with_prediction = torch.cat([X_eval, predictions_hardened], axis=1)
    
    # measure the demogprahic parity distance on the resulting joint data
    demographic_parity = demographic_parity_distance(
        data=joint_evaluation_data_with_prediction,
        target_feature=target_feature,
        protected_feature=protected_feature,
        dataset=dataset,
        operation='mean'
    )

    return demographic_parity


def downstream_equalized_odds(X_train, y_train, X_eval, y_eval, model, dataset, protected_feature, 
                              target_feature=None, desired_outcome=None, lr=0.01, batch_size=512, num_epochs=10):
    """
    This function takes a differentiable model, a training dataset, and an evaluation dataset, and computes the
    equalized odds distance of the trained model on the evaluation dataset with respect to a given feature
    in a differentiable manner.

    :param X_train: (torch.tensor) Training features one-hot encoded.
    :param y_train: (torch.tensor of torch.long) Training labels, ordinal encoded.
    :param X_eval: (torch.tensor) Evaluation features one-hot encoded.
    :param y_eval: (torch.tensor of torch.long) Evaluation labels, ordinal encoded.
    :param model: (torch.nn.Module) Instantiated differentiable torch classification model. Make sure that the model is 
        supported by monkey patcing.
    :param dataset: (BaseDataset) The instantiated dataset object containing the necessary information for the data.
    :param protected_feature: (str) The name of the protected feature.
    :param target_feature: (str) The name of the target (label) feature.
    :param desired_outcome: (str) The desired outcome with respect to which the TPR and the FPR are calculated.
    :param lr: (float) Learning rate for training the model with gradient descent.
    :param batch_size: (int) Batch size for training.
    :param num_epochs: (int) Number of epochs for training.
    :return: (torch.tensor) The equalized odds distance of the trained model on the evaluationd data.
    """
    if target_feature is None:
        target_feature = dataset.label
    
    if desired_outcome is None:
        desired_outcome = dataset.features[target_feature][-1]
    
    # monkey patch the model to be able to do backrpop through it
    monkey_model = MetaMonkey(model)
    
    # train the model
    criterion = torch.nn.CrossEntropyLoss() 
    train_model_differentiable_monkey(
        Xtr=X_train, 
        ytr=y_train, 
        monkey_model=monkey_model, 
        criterion=criterion, 
        lr=lr, 
        batch_size=batch_size, 
        num_epochs=num_epochs
    )
    
    # make predictions on the evaluation data
    predictions = monkey_model(X_eval, monkey_model.parameters)
    predictions_hardened = straight_through_softmax(predictions)
    joint_evaluation_data_with_prediction = torch.cat([X_eval, predictions_hardened], axis=1)

    # convert the true label to one-hot
    y_eval_one_hot = torch.zeros((len(y_eval), len(dataset.features[target_feature]))).to(y_eval.device)
    y_eval_one_hot[np.arange(len(y_eval)), y_eval] = 1.
    
    # measure the demogprahic parity distance on the resulting joint data
    equalized_odds = equalized_odds_distance(
        data=joint_evaluation_data_with_prediction,
        true_labels=y_eval_one_hot,
        target_feature=target_feature,
        protected_feature=protected_feature,
        desired_outcome=desired_outcome,
        dataset=dataset,
        operation='mean'
    )

    return equalized_odds


def downstream_equality_of_opportunity(X_train, y_train, X_eval, y_eval, model, dataset, protected_feature, 
                                       target_feature=None, desired_outcome=None, lr=0.01, batch_size=512, num_epochs=10):
    """
    This function takes a differentiable model, a training dataset, and an evaluation dataset, and computes the
    equality of opportunity distance of the trained model on the evaluation dataset with respect to a given feature
    in a differentiable manner.

    :param X_train: (torch.tensor) Training features one-hot encoded.
    :param y_train: (torch.tensor of torch.long) Training labels, ordinal encoded.
    :param X_eval: (torch.tensor) Evaluation features one-hot encoded.
    :param y_eval: (torch.tensor of torch.long) Evaluation labels, ordinal encoded.
    :param model: (torch.nn.Module) Instantiated differentiable torch classification model. Make sure that the model is 
        supported by monkey patcing.
    :param dataset: (BaseDataset) The instantiated dataset object containing the necessary information for the data.
    :param protected_feature: (str) The name of the protected feature.
    :param target_feature: (str) The name of the target (label) feature.
    :param desired_outcome: (str) The desired outcome with respect to which the TPR and the FPR are calculated.
    :param lr: (float) Learning rate for training the model with gradient descent.
    :param batch_size: (int) Batch size for training.
    :param num_epochs: (int) Number of epochs for training.
    :return: (torch.tensor) The equality of opportunity distance of the trained model on the evaluationd data.
    """
    if target_feature is None:
        target_feature = dataset.label
    
    if desired_outcome is None:
        desired_outcome = dataset.features[target_feature][-1]
    
    # monkey patch the model to be able to do backrpop through it
    monkey_model = MetaMonkey(model)
    
    # train the model
    criterion = torch.nn.CrossEntropyLoss() 
    train_model_differentiable_monkey(
        Xtr=X_train, 
        ytr=y_train, 
        monkey_model=monkey_model, 
        criterion=criterion, 
        lr=lr, 
        batch_size=batch_size, 
        num_epochs=num_epochs
    )
    
    # make predictions on the evaluation data
    predictions = monkey_model(X_eval, monkey_model.parameters)
    predictions_hardened = straight_through_softmax(predictions)
    joint_evaluation_data_with_prediction = torch.cat([X_eval, predictions_hardened], axis=1)

    # convert the true label to one-hot
    y_eval_one_hot = torch.zeros((len(y_eval), len(dataset.features[target_feature]))).to(y_eval.device)
    y_eval_one_hot[np.arange(len(y_eval)), y_eval] = 1.
    
    # measure the demogprahic parity distance on the resulting joint data
    equality_of_opportunity = equality_of_opportunity_distance(
        data=joint_evaluation_data_with_prediction,
        true_labels=y_eval_one_hot,
        target_feature=target_feature,
        protected_feature=protected_feature,
        desired_outcome=desired_outcome,
        dataset=dataset,
        operation='mean'
    )

    return equality_of_opportunity
