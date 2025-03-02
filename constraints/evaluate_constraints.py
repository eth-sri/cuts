import torch
import numpy as np
from .logic import create_mask_from_parsed
from .parsing import ConstraintProgramParser
from .compile_constraints import ConstraintCompiler
from .statistical import expectation, variance, standard_deviation, entropy
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from utils import demographic_parity_distance, equalized_odds_distance, equality_of_opportunity_distance


class ConstraintEvaluator:

    def __init__(self, program, dataset, base_data, program_arguments=None, user_custom_functions=None, random_state=42, classifiers=['xgb'], device='cuda'):

        """
        Evaluates synthetic data with respect to a constraint program.
        """

        self.parser = ConstraintProgramParser(features=dataset.features)
        self.program_arguments = program_arguments
        self.parsed_program = self.parser.parse_constraint_program(program, self.program_arguments)
        
        self.dataset = dataset
        self.base_data = base_data
        self.random_state = random_state
        self.device = device
        self.chosen_classifiers = classifiers

        self.user_custom_functions = {} if user_custom_functions is None else user_custom_functions

        self.command_type_evaluators = {
                'row constraint': self._row_constraint_evaluator,
                'implication': self._implication_evaluator,
                'statistical': self._statistical_evaluator,
                'statistical_logical': self._statistical_logical_evaluator,
                'utility': self._utility_evaluator,
                'bias': self._bias_evaluator,
                'user custom': self._user_custom_evaluator,
            }

        self.available_classifiers = {
            'logreg': (LogisticRegression, {'random_state': self.random_state}),
            'svm': (SVC, {'random_state': self.random_state}),
            'rf': (RandomForestClassifier, {'random_state': self.random_state}),
            'xgb': (XGBClassifier, {'verbosity': 0, 'random_state': self.random_state}),
        }

        self.statistical_operators = {
            'E': expectation,
            'Var': variance,
            'STD': standard_deviation,
            'H': entropy 
        }

    def evaluate_constraints(self, syn_data):
        """
        Main method of this object, evaluated the given synthetic data on the satisfaction of the constraints.

        :param syn_data: (torch.tensor) The synthetic data that is to be evaluated.
        :return: (list[dict]) For each constraint the corresponding evaluation score. 
        """
        evaluation_data = []

        for constraint in self.parsed_program:

            # DP constraint is not handled here
            if constraint['command_type'] == 'differential privacy':
                continue
            
            command_type_evaluator = 'statistical_logical' if constraint['command_type'] == 'statistical' and constraint['action'] == 'enforce' else constraint['command_type']
            constraint_score = self.command_type_evaluators[command_type_evaluator](syn_data, constraint['parsed_command'])
            evaluation_data.append({'score': constraint_score, 'parsed_constraint': constraint['original_command'].strip()})
        
        return evaluation_data

    def _get_classifier(self, classifier_name):
        """
        Helper method to return an instantiated classifier.

        :param classifier_name: (str) The name of the classifier that we want to return.
        :return: (sklearn.BaseEstimator) A classifier to use.
        """
        return self.available_classifiers[classifier_name][0](**self.available_classifiers[classifier_name][1])

    def _row_constraint_evaluator(self, syn_data, parsed_row_constraint_command):
        """
        Evaluates row constraints, returning the constraint satisfaction rate: #rows meeting the constraint / #all rows.

        :param syn_data: (torch.tensor) The synthetic data to be evaluated.
        :param parsed_row_constraint_command: (list) The parsed row command.
        :return: (float) The constraint satisfaction rate.
        """
        conforming_rows = self._recursive_row_constraint_selector(syn_data, parsed_row_constraint_command, compensate_redundancy=False)
        constraint_satisfaction_rate = (conforming_rows.sum() / syn_data.size()[0]).item()

        return constraint_satisfaction_rate

    def _recursive_row_constraint_selector(self, syn_data, parsed_row_constraint_command, compensate_redundancy=False):
        """
        Recursive method to convert a parsed row constraint expression into a binary mask over the rows
        of the synthetic dataset, where there is a non-zero entry at each row where the expression is met.

        :param syn_data: (torch.tensor) The synthetic data over which we evaluate the expression.
        :param parsed_row_constraint_command: (list) The parsed row constraint command that we evaluate recursively. The nested lists
            should describe the operation tree.
        :param compensate_redundancy: (bool) Toggle to compensate for the redundancy caused by chained OR expressions on the same operation
            precedence level. This is achieved by applying the inclusion-exclusion principle on a binary tree. Note that for this option
            to work correctly, the opration tree in parsed_row_constraint_command has to be binary.
        :return: (torch.tensor) The resulting binary mask over the rows of the data.
        """
        current_row_masks = []
        operation = None

        for token in parsed_row_constraint_command:
            
            # if we arrived at a leaf expression of ['feature', 'operator', 'constant'], then we create the mask
            # and produce the row mask, appending to all row masks of all leaves of the same parent
            if isinstance(token, list) and isinstance(token[0], str):
                assert len(token) == 3, f'Unable to compile leaf-node {token}'
                mask = create_mask_from_parsed(syn_data, self.dataset, *token)
                row_mask = syn_data @ mask.T
                current_row_masks.append(row_mask)
            
            # if the token is and operation, we record it
            elif token == 'OR' or token == 'AND':
                operation = token
            
            # recurse deeper in the tree
            elif isinstance(token, list):
                row_mask = self._recursive_row_constraint_selector(syn_data, token, compensate_redundancy=compensate_redundancy)
                current_row_masks.append(row_mask)
            
            # unknown element, raise error
            else:
                raise RuntimeError(f'Unable to handle token {token}')
            
        # resolve the operation
        if operation == 'OR':
            final_row_mask = torch.zeros(syn_data.size()[0]).to(self.device)
            redundancy_mask = torch.ones(syn_data.size()[0]).to(self.device)
            for row_mask in current_row_masks:
                final_row_mask += row_mask
                redundancy_mask *= row_mask
            if compensate_redundancy:
                final_row_mask -= redundancy_mask

        # the AND operator also forms the basecase of just passing a single row mask as identity (the same could be achieved with OR as well)
        else:
            final_row_mask = torch.ones(syn_data.size()[0]).to(self.device)
            for row_mask in current_row_masks:
                final_row_mask *= row_mask
        
        # difference to training compiler
        final_row_mask.clamp_(min=0.0, max=1.0)
        
        return final_row_mask

    def _implication_evaluator(self, syn_data, parsed_implication_command):
        """
        Evaluates an implication, returning the constraint satisfaction rate: 1 - #violations / #antecdent applies.

        :param syn_data: (torch.tensor) The synthetic data to be evaluated.
        :param parsed_implication_command: (list) The parsed implication command to be evaluted.
        :return: (float) The resulting constraint satisfaction rate.
        """
        # get the masks, not that this is not differentiable anymore, as it is clamped to 0, 1
        antecedent_row_mask = self._recursive_row_constraint_selector(syn_data, parsed_implication_command['antecedent'], compensate_redundancy=True)
        neg_consequent_row_mask = self._recursive_row_constraint_selector(syn_data, parsed_implication_command['neg_consequent'])

        # count the violations and calculate the constraint satisfaction rate
        num_violations = (antecedent_row_mask * neg_consequent_row_mask).sum().item()
        constraint_satisfaction_rate = 1. - num_violations / max(antecedent_row_mask.sum().item(), 1.)

        return constraint_satisfaction_rate

    def _statistical_logical_evaluator(self, syn_data, parsed_statistical_logical_command):
        """
        Compiles composite statistical expressions that are to be enforced over the dataset.

        :param syn_data: (torch.tensor) The synthetic data produced by the synthesizer currently.
        :param parsed_statistical_logical_command: (list) Logical-Statistical expression parsed into an operation tree.
        :return: (torch.tensor) Result of the evaluated logical-statistical expression.
        """
        operation = None
        values = []

        for token in parsed_statistical_logical_command:
            
            # we arrived at a comparison of two statistical expressions, compile them and then apply dl2-like operation translation
            if len(token) == 3 and token[1] in ['>', '<', '>=', '<=', '!=', '==']:
                left_operand = self._statistical_evaluator(syn_data, [token[0]])
                right_operand = self._statistical_evaluator(syn_data, [token[2]])

                if token[1] == '>=':
                    violation_score = max(0, right_operand - left_operand)
                elif token[1] == '<=':
                    violation_score = max(0, left_operand - right_operand)
                elif token[1] == '>':
                    violation_score = max(0, right_operand - left_operand) + int(right_operand == left_operand)
                elif token[1] == '<':
                    violation_score = max(0, left_operand - right_operand) + int(right_operand == left_operand)
                elif token[1] == '==':
                    violation_score = (right_operand - left_operand)**2
                elif token[1] == '!=':
                    violation_score = int(left_operand == right_operand)
                else:
                    raise ValueError(f'Unknown constraint')

                values.append(violation_score)

            elif token in ['OR', 'AND']:
                operation = token
            
            else:
                values.append(self._statistical_logical_evaluator(syn_data, token))
            
        if operation is None:
            if len(values) > 1:
                raise RuntimeError(f'Orphanaged operands in the parsed command: {parsed_statistical_logical_command}')
            result = values[0]
        elif operation == 'OR':
            result = 1.
            for val in values:
                result *= val
        elif operation == 'AND':
            result = 0.
            for val in values:
                result += val
        else:
            raise RuntimeError(f'Invalid operator: {operation}')
        
        return result

    def _statistical_evaluator(self, syn_data, parsed_statistical_command):
        """
        Recursive evaluator for a complex statistical expression.

        :param syn_data: (torch.tensor) The synthetic data produced by the synthesizer currently.
        :param parsed_statistical_command: (list) Statistical expression parsed into an operation tree.
        :return: Result of the evaluated statistical expression.
        """
        result = torch.tensor([0.0]).to(self.device)
        preceeding_operation = '+'

        for token in parsed_statistical_command:

            # evaluate the statistical expectation
            if isinstance(token, list) and len(token) == 3 and token[0] in list(self.statistical_operators.keys()):
                term_val = self._single_statistical_expression_evaluator(syn_data, token)
                result = self._apply_operation_from_string_operator(result, term_val, preceeding_operation)
            
            # go deeper in the recursion
            elif isinstance(token, list):
                term_val = self._statistical_evaluator(syn_data, token)
                result = self._apply_operation_from_string_operator(result, term_val, preceeding_operation)
            
            # record the operation
            elif token in ['*', '/', '+', '-']:
                preceeding_operation = token
            
            # constants
            else:
                term_val = float(token)
                result = self._apply_operation_from_string_operator(result, term_val, preceeding_operation)

        return result.item()

    def _apply_operation_from_string_operator(self, in1, in2, operation):
        """
        Very simple method to translate an operation in string to actual python between two tensors. I am sure that there is a
        more elegant solution to this.

        :param in1: (torch.tensor or float) Left operand.
        :param in2: (torch.tensor or float) Right operand.
        :param operation: (str) The operation.
        :return: (torch.tensor or float) The result of the executed operation.
        """
        assert operation in ['*', '/', '+', '-'], f'{operation} operation unknown'
        if operation == '*':
            result = in1 * in2
        elif operation == '/':
            result = in1 / (in2 + 1e-7)
        elif operation == '+':
            result = in1 + in2
        else:
            result = in1 - in2
        return result

    def _single_statistical_expression_evaluator(self, syn_data, parsed_single_statistical_expression):
        """
        Takes a single statistical expression and returns the resulting value.

        :param syn_data: (torch.tensor) The synthetic data to calcualte the operation on.
        :param parsed_single_statistical_expression: (list) A list of length three, where the first entry is the operator, the second the
            expression to be evaluated under the statistical operation, and the third the condition. The latter two are parsed into operation trees.
        :return: (torch.tensor) The resulting value from the statistical operation.
        """
        operator, random_variable, condition = parsed_single_statistical_expression

        assert operator in list(self.statistical_operators.keys()), f'{operator} is an unknown statistical operator. Available are {list(self.statistical_operators.keys())}'

        # get the row mask for the condition
        if condition == 'None':
            condition_row_mask = torch.ones(syn_data.size()[0]).to(self.device)
        else:
            binary_tree_condition = ConstraintProgramParser.binarize_first_order_logic_operation_tree(condition)
            condition_row_mask = self._recursive_row_constraint_selector(syn_data, [binary_tree_condition], compensate_redundancy=True)
        
        # extract the random variable as a lambda function
        lambda_function, involved_features = ConstraintProgramParser.parsed_expression_to_lambda_function(random_variable, self.dataset.features)

        # evaluate the conditional statistical expression
        result = self.statistical_operators[operator](syn_data, self.dataset, involved_features, lambda_function, condition_row_mask)

        return result
    
    def _utility_evaluator(self, syn_data, parsed_utility_command):
        """
        Evaluates the downstream utility command by training an XGBoost on the defined feature space and labels, and
        reporting its accuracy, balanced accuracy, and F1 score.

        :param syn_data: (torch.tensor) The synthetic data to be evaluated.
        :param parsed_utility_command: (dict) Dictionary containing the name of the downstream measure 'function_name', and 
            the keyowrd arguments of the given measure.
        :return: (list) The accuracy, balanced accuracy, and the F1 score of the resulting model on the test data.
        """
        if parsed_utility_command['kwargs']['features'] == 'all':
            predict_on_features = [feature_name for feature_name in self.dataset.features.keys() if feature_name != parsed_utility_command['kwargs']['target']]
        else:
            predict_on_features = parsed_utility_command['kwargs']['features']

        X_train, y_train = ConstraintCompiler.prepare_data(syn_data, predict_on_features, parsed_utility_command['kwargs']['target'], self.dataset, self.device)
        X_eval, y_eval = ConstraintCompiler.prepare_data(self.base_data, predict_on_features, parsed_utility_command['kwargs']['target'], self.dataset, self.device)
        
        # avoid encoding error
        y_train = ConstraintEvaluator.handle_missing_classes_in_training_data(y_train, self.dataset.features[parsed_utility_command['kwargs']['target']])
        
        classifier = self._get_classifier(classifier_name)
        classifier.fit(X_train.cpu().numpy(), y_train.cpu().numpy().astype(int))
        prediction = classifier.predict(X_eval.cpu().numpy())

        acc, bac, f1 = accuracy_score(y_eval.cpu().numpy(), prediction), balanced_accuracy_score(y_eval.cpu().numpy(), prediction), f1_score(y_eval.cpu().numpy(), prediction, average='micro')

        return [acc, bac, f1]
    
    def _bias_evaluator(self, syn_data, parsed_bias_command):
        """
        Trains an XGBoost on the synthetic data and measures its bias on the test data.

        :param syn_data: (torch.tensor) The synthetic data to be evaluated.
        :param parsed_bias_command: (dict) The parsed bias command.
        :return: (float) The bias score.
        """
        X_train, y_train = ConstraintCompiler.prepare_data(syn_data, list(self.dataset.train_features.keys()), self.dataset.label, self.dataset, self.device)
        X_eval, y_eval = ConstraintCompiler.prepare_data(self.base_data, list(self.dataset.train_features.keys()), self.dataset.label, self.dataset, self.device)

        # avoid encoding error
        y_train = ConstraintEvaluator.handle_missing_classes_in_training_data(y_train, self.dataset.features[self.dataset.label])

        bias_scores = []
        for classifier_name in self.chosen_classifiers:
        
            classifier = self._get_classifier(classifier_name)
            classifier.fit(X_train.cpu().numpy(), y_train.cpu().numpy().astype(int))
            prediction = classifier.predict(X_eval.cpu().numpy())

            # prediction = (prediction > 0.5).astype(int)

            # go back to one hot
            one_hot_eval = torch.zeros((len(y_eval), len(self.dataset.full_one_hot_index_map[self.dataset.label]))).to(self.device)
            one_hot_eval[np.arange(len(y_eval)), y_eval] = 1.
            one_hot_prediction = torch.zeros((len(prediction), len(self.dataset.full_one_hot_index_map[self.dataset.label]))).to(self.device)
            one_hot_prediction[np.arange(len(prediction)), prediction] = 1.

            # attach this to the data
            joint_eval_data = torch.cat([X_eval, one_hot_prediction], axis=1)

            # extend the dict
            if 'desired_outcome' not in parsed_bias_command['kwargs']: parsed_bias_command['kwargs']['desired_outcome'] = self.dataset.features[self.dataset.label][-1]

            if parsed_bias_command['function_name'] == 'demographic_parity':
                bias_score = demographic_parity_distance(
                    data=joint_eval_data,
                    target_feature=self.dataset.label,
                    protected_feature=parsed_bias_command['kwargs']['protected'],
                    dataset=self.dataset,
                    operation='max'
                )
            
            elif parsed_bias_command['function_name'] == 'equality_of_opportunity':
                bias_score = equality_of_opportunity_distance(
                    data=joint_eval_data,
                    true_labels=one_hot_eval,
                    target_feature=self.dataset.label,
                    protected_feature=parsed_bias_command['kwargs']['protected'],
                    desired_outcome=parsed_bias_command['kwargs']['desired_outcome'],
                    dataset=self.dataset,
                    operation='max'
                )
            
            elif parsed_bias_command['function_name'] == 'equalized_odds':
                bias_score = equalized_odds_distance(
                    data=joint_eval_data,
                    true_labels=one_hot_eval,
                    target_feature=self.dataset.label,
                    protected_feature=parsed_bias_command['kwargs']['protected'],
                    desired_outcome=parsed_bias_command['kwargs']['desired_outcome'],
                    dataset=self.dataset,
                    operation='max'
                )

            else:
                fm = parsed_bias_command['function_name']
                raise NotImplementedError(f'Bias measure {fm} is not implemented.')
            bias_scores.append(bias_score.item())

        return bias_scores

    def _user_custom_evaluator(self, syn_data, parsed_user_custom_command):
        """
        Takes a parsed custom user command that involves the calling of a function that was defined by the user in the python scope 
        of the original classifier and passed to this object. It executes the user function to calculate the result.

        :param syn_data: (torch.tensor) The synthetic data generated by the current generator.
        :param parsed_user_custom_command: (dict) Dictionary containing the name of the user defined function 'function_name', and 
            the keyowrd arguments of the given function.
        :return: (torch.tensor) The differentiable user-defined score.
        """
        user_custom_function_score = self.user_custom_functions[parsed_user_custom_command['function_name']](
                                        syn_data=syn_data, 
                                        base_data=self.base_data, 
                                        dataset=self.dataset, 
                                        *parsed_user_custom_command['kwargs']
                                    )
        return user_custom_function_score

    @staticmethod
    def handle_missing_classes_in_training_data(y_train, feature_domain):
        """
        Makes sure that the xgboost model used for evaluation does not infer less classes than what is present in the
        evaluation data. Note that although this function artifically flips some labels, it is already bad enough that
        it has to be called, meaning already that the train data did not cover the whole range of the test data.

        :param y_train: (torch.tensor) The training labels.
        :param feature_domain: (list) The domain of the target feature.
        :return: (torch.tensor) The fixed labels.
        """
        # avoid encoding error
        train_num_classes, true_num_classes = len(np.unique(y_train.cpu().numpy())), len(feature_domain)
        difference = true_num_classes - train_num_classes
        if difference > 0:
            random_indices_to_flip = np.random.randint(0, len(y_train)-1, true_num_classes)
            for label, index in enumerate(random_indices_to_flip):
                y_train[index] = label
        return y_train
