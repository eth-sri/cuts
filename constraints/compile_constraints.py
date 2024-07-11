import torch
import numpy as np
from itertools import chain
from classification_models import LogReg
from .downstream import downstream_demographic_parity, downstream_prediction, downstream_equality_of_opportunity, downstream_equalized_odds
from .logic import create_mask_from_parsed
from .parsing import ConstraintProgramParser
from .statistical import expectation, variance, standard_deviation, entropy
from utils import dl2_geq, dl2_neq


class ConstraintCompiler:

    """
    Compiles the program into a pytorch regularizer.
    """

    def __init__(self, program, dataset, base_data, program_arguments=None, user_custom_functions=None, device='cuda'):

        self.parser = ConstraintProgramParser(features=dataset.features)
        self.program_arguments = program_arguments
        self.parsed_program = self.parser.parse_constraint_program(program, self.program_arguments)
        self.dataset = dataset
        self.base_data = base_data
        self.device = device

        self.user_custom_functions = {} if user_custom_functions is None else user_custom_functions

        self.command_type_compilers = {
                'row constraint': self._row_constraint_compiler,
                'implication': self._implication_compiler,
                'statistical': self._statistical_compiler,
                'statistical_logical': self._statistical_logical_compiler,
                'utility': self._utility_compiler,
                'bias': self._bias_compiler,
                'user custom': self._user_custom_compiler,
            }

        self.default_params = {
            'row constraint': 0.01,
            'implication': 0.01,
            'statistical': 0.01,
            'utility': 0.01,
            'bias': 0.01,
            'user custom': 0.01,
        }

        self.bias_measures = {
            'demographic_parity': downstream_demographic_parity,
            'equality_of_opportunity': downstream_equality_of_opportunity,
            'equalized_odds': downstream_equalized_odds
        }

        self.statistical_operators = {
            'E': expectation,
            'Var': variance,
            'STD': standard_deviation,
            'H': entropy 
        }
    
    def add_user_custom_function(self, func):
        """
        A setter function to add user-defined python functions to the evaluation.

        :param func: (callable) The user defined function where the first three positional arguments are:
            syn_data, base_data, dataset, and the rest are keyword arguments. Note that this function will not have access
            to the internal function of this object.
        :return: self
        """
        self.user_custom_functions[func.__name__] = func
        return self

    def compile_regularizer(self, syn_data):
        """
        The main method of this object that returns the regularization/penalty term applied on the loss of the synthesizer
        to encourage that the constraint and other specifications are eventually met.

        :param syn_data: (torch.tensor) The synthetic data produced by the synthesizer currently.
        :return: (torch.tensor) The regularization term.
        """
        regularizer = torch.tensor([0.0], device=self.device)

        for constraint in self.parsed_program:

            # DP constraint is not handled here
            if constraint['command_type'] == 'differential privacy':
                continue
            
            # if the constaint param is None, we resort to some default param
            if constraint['param'] is None:
                constraint['param'] = self.default_params[constraint['command_type']]
            
            # make sure that the constraint parameter is pointing the optimizer to the right direction
            maximization_factor = 1. if constraint['command_type'] == 'utility' else -1. 
            param = maximization_factor * constraint['param'] if constraint['action'] == 'maximize' else -1 * maximization_factor * constraint['param']

            command_type_compiler = 'statistical_logical' if constraint['command_type'] == 'statistical' and constraint['action'] == 'enforce' else constraint['command_type']
            regularizer += param * self.command_type_compilers[command_type_compiler](syn_data, constraint['parsed_command'])

        return regularizer

    def _row_constraint_compiler(self, syn_data, parsed_row_constraint_command):
        """
        Wrapper method to convert a parsed row constraint that is to be enforced over the data into the corresponding regularization
        term. This is achieved by first negating the expression, counting the number of violating rows, and finally summing this up
        such that we can later minimize to encourage the satsifcation of the original constraint.

        :param syn_data: (torch.tensor) The synthetic data produced by the synthesizer currently.
        :param parsed_row_constraint: (list) Parsed row constraint expression tree.
        :return: (torch.tensor) The resulting regularizer.
        """
        
        # negate the expression as we are trying to minimize the number of violations
        negated_row_constraint_command = ConstraintProgramParser.negate_parsed_logical_expression(parsed_row_constraint_command)

        # now we get the full map of violation the length of the syn_data
        violating_rows = self._recursive_row_constraint_compiler(syn_data, negated_row_constraint_command)

        row_constraint_regularizer = violating_rows.sum()

        return row_constraint_regularizer
    
    def _recursive_row_constraint_compiler(self, syn_data, parsed_row_constraint_command, compensate_redundancy=False):
        """
        Recursive method to convert a parsed row constraint expression into a mask over the rows
        of the synthetic dataset, where there is a non-zero entry at each row where the expression is met.

        :param syn_data: (torch.tensor) The synthetic data over which we evaluate the expression.
        :param parsed_row_constraint_command: (list) The parsed row constraint command that we evaluate recursively. The nested lists
            should describe the operation tree.
        :param compensate_redundancy: (bool) Toggle to compensate for the redundancy caused by chained OR expressions on the same operation
            precedence level. This is achieved by applying the inclusion-exclusion principle on a binary tree. Note that for this option
            to work correctly, the opration tree in parsed_row_constraint_command has to be binary.
        :return: (torch.tensor) The resulting mask over the rows of the data. Its sum is lower-bounded by the number of rows that meet
            the condition in syn_data.
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
                row_mask = self._recursive_row_constraint_compiler(syn_data, token, compensate_redundancy=compensate_redundancy)
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
        
        return final_row_mask    

    def _implication_compiler(self, syn_data, parsed_implication_command):
        """
        Takes a compiled implication command and returns a regularizer that is proportional to the number of violation of the implication
        in the current dataset. The basic principle is to turn the enforcmenet of an implication A -> B into the minimization of the
        occurrence of A AND NOT B.

        :param syn_data: (torch.tensor) The synthetic data produced by the synthesizer currently.
        :param parsed_implication_command: (dict) Dictionary containing the binary tree parsed antecedent (A) and the parsed negated consequent (NOT B).
        :return: (torch.tensor) The regularizer in proportion to the amount of violations.
        """
        antecedent_row_mask = self._recursive_row_constraint_compiler(syn_data, parsed_implication_command['antecedent'], compensate_redundancy=True)
        neg_consequent_row_mask = self._recursive_row_constraint_compiler(syn_data, parsed_implication_command['neg_consequent'])

        implication_regularizer = (antecedent_row_mask * neg_consequent_row_mask).sum()

        return implication_regularizer
    
    def _statistical_logical_compiler(self, syn_data, parsed_statistical_logical_command):
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
                left_operand = self._statistical_compiler(syn_data, [token[0]])
                right_operand = self._statistical_compiler(syn_data, [token[2]])

                if token[1] == '>=':
                    violation_score = dl2_geq(left_operand, right_operand)
                elif token[1] == '<=':
                    violation_score = dl2_geq(right_operand, left_operand)
                elif token[1] == '>':
                    violation_score = (dl2_geq(left_operand, right_operand) + dl2_neq(left_operand, right_operand))
                elif token[1] == '<':
                    violation_score = (dl2_geq(right_operand, left_operand) + dl2_neq(left_operand, right_operand))
                elif token[1] == '==':
                    violation_score = (right_operand - left_operand).pow(2)
                elif token[1] == '!=':
                    violation_score = dl2_neq(right_operand, left_operand)
                else:
                    raise ValueError(f'Unknown constraint')

                values.append(violation_score)

            elif token in ['OR', 'AND']:
                operation = token
            
            else:
                values.append(self._statistical_logical_compiler(syn_data, token))
            
        if operation is None:
            if len(values) > 1:
                raise RuntimeError(f'Orphanaged operands in the parsed command: {parsed_statistical_logical_command}')
            result = values[0]
        elif operation == 'OR':
            result = torch.tensor([1.], device=self.device)
            for val in values:
                result *= val
        elif operation == 'AND':
            result = torch.tensor([0.], device=self.device)
            for val in values:
                result += val
        else:
            raise RuntimeError(f'Invalid operator: {operation}')
        
        return result

    def _statistical_compiler(self, syn_data, parsed_statistical_command):
        """
        Recursive compiler for a complex statistical expression.

        :param syn_data: (torch.tensor) The synthetic data produced by the synthesizer currently.
        :param parsed_statistical_command: (list) Statistical expression parsed into an operation tree.
        :return: (torch.tensor) Result of the evaluated statistical expression.
        """
        result = torch.tensor([0.0]).to(self.device)
        preceeding_operation = '+'

        for token in parsed_statistical_command:

            # evaluate the statistical expectation
            if isinstance(token, list) and len(token) == 3 and token[0] in list(self.statistical_operators.keys()):
                term_val = self._single_statistical_expression_compiler(syn_data, token)
                result = self._apply_operation_from_string_operator(result, term_val, preceeding_operation)
            
            # go deeper in the recursion
            elif isinstance(token, list):
                term_val = self._statistical_compiler(syn_data, token)
                result = self._apply_operation_from_string_operator(result, term_val, preceeding_operation)
            
            # record the operation
            elif token in ['*', '/', '+', '-']:
                preceeding_operation = token
            
            # constants
            else:
                term_val = float(token)
                result = self._apply_operation_from_string_operator(result, term_val, preceeding_operation)

        return result

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

    def _single_statistical_expression_compiler(self, syn_data, parsed_single_statistical_expression):
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
            condition_row_mask = self._recursive_row_constraint_compiler(syn_data, [binary_tree_condition], compensate_redundancy=True)
        
        # convert to lambda function
        lambda_function, involved_features = ConstraintProgramParser.parsed_expression_to_lambda_function(random_variable, self.dataset.features)

        # evaluate the conditional statistical expression
        result = self.statistical_operators[operator](syn_data, self.dataset, involved_features, lambda_function, condition_row_mask)

        return result
    
    def _utility_compiler(self, syn_data, parsed_utility_command):
        """
        Takes a parsed utility command and returns the differentiable utility score to be optimized.

        :param syn_data: (torch.tensor) The synthetic data generated by the current generator.
        :param parsed_utility_command: (dict) Dictionary containing the name of the downstream measure 'function_name', and 
            the keyowrd arguments of the given measure.
        :return: (torch.tensor) Differentiable utility score on the defined feature set.
        """

        if parsed_utility_command['kwargs']['features'] == 'all':
            predict_on_features = [feature_name for feature_name in self.dataset.features.keys() if feature_name != parsed_utility_command['kwargs']['target']]
        else:
            predict_on_features = parsed_utility_command['kwargs']['features']

        X_train, y_train = ConstraintCompiler.prepare_data(syn_data, predict_on_features, parsed_utility_command['kwargs']['target'], self.dataset, self.device)
        X_eval, y_eval = ConstraintCompiler.prepare_data(self.base_data, predict_on_features, parsed_utility_command['kwargs']['target'], self.dataset, self.device)

        default_model_training_specs = ConstraintCompiler.get_default_model_training_specs(input_dim=X_train.size()[1], output_dim=torch.max(y_eval).item() + 1, device=self.device)
        lr = default_model_training_specs['lr'] if 'lr' not in parsed_utility_command['kwargs'] else parsed_utility_command['kwargs']['lr']
        batch_size =  default_model_training_specs['batch_size'] if 'batch_size' not in parsed_utility_command['kwargs'] else parsed_utility_command['kwargs']['batch_size']
        num_epochs =  default_model_training_specs['num_epochs'] if 'num_epochs' not in parsed_utility_command['kwargs'] else parsed_utility_command['kwargs']['num_epochs']

        downstream_utility_score = downstream_prediction(
            X_train=X_train, 
            y_train=y_train, 
            X_eval=X_eval, 
            y_eval=y_eval, 
            model=default_model_training_specs['model'], 
            lr=float(lr), 
            batch_size=int(batch_size), 
            num_epochs=int(num_epochs)
        )

        return downstream_utility_score

    def _bias_compiler(self, syn_data, parsed_bias_command):
        """
        Takes a parsed bias command and returns the differentiable bias score to be optimized.

        :param syn_data: (torch.tensor) The synthetic data generated by the current generator.
        :param parsed_bias_command: (dict) Dictionary containing the name of the bias measure 'function_name', and 
            the keyowrd arguments of the given measure.
        :return: (torch.tensor) The differentiable bias score.
        """
        
        X_train, y_train = ConstraintCompiler.prepare_data(syn_data, list(self.dataset.train_features.keys()), self.dataset.label, self.dataset, self.device)
        X_eval, y_eval = ConstraintCompiler.prepare_data(self.base_data, list(self.dataset.train_features.keys()), self.dataset.label, self.dataset, self.device)

        default_model_training_specs = ConstraintCompiler.get_default_model_training_specs(input_dim=X_train.size()[1], output_dim=torch.max(y_eval).item() + 1, device=self.device)
        lr = default_model_training_specs['lr'] if 'lr' not in parsed_bias_command['kwargs'] else parsed_bias_command['kwargs']['lr']
        batch_size =  default_model_training_specs['batch_size'] if 'batch_size' not in parsed_bias_command['kwargs'] else parsed_bias_command['kwargs']['batch_size']
        num_epochs =  default_model_training_specs['num_epochs'] if 'num_epochs' not in parsed_bias_command['kwargs'] else parsed_bias_command['kwargs']['num_epochs']

        if 'desired_outcome' not in parsed_bias_command['kwargs']: parsed_bias_command['kwargs']['desired_outcome'] = None

        if parsed_bias_command['function_name'] == 'demographic_parity':

            bias_score = downstream_demographic_parity(
                X_train=X_train,
                y_train=y_train,
                X_eval=X_eval,
                y_eval=y_eval,
                model=default_model_training_specs['model'],
                dataset=self.dataset,
                protected_feature=parsed_bias_command['kwargs']['protected'],
                target_feature=self.dataset.label,
                lr=float(lr),
                batch_size=int(batch_size),
                num_epochs=int(num_epochs)
            )
        
        elif parsed_bias_command['function_name'] == 'equality_of_opportunity':

            bias_score = downstream_equality_of_opportunity(
                X_train=X_train,
                y_train=y_train,
                X_eval=X_eval,
                y_eval=y_eval,
                model=default_model_training_specs['model'],
                dataset=self.dataset,
                protected_feature=parsed_bias_command['kwargs']['protected'],
                target_feature=self.dataset.label,
                desired_outcome=parsed_bias_command['kwargs']['desired_outcome'],
                lr=float(lr),
                batch_size=int(batch_size),
                num_epochs=int(num_epochs)
            )
        
        elif parsed_bias_command['function_name'] == 'equalized_odds':

            bias_score = downstream_equalized_odds(
                X_train=X_train,
                y_train=y_train,
                X_eval=X_eval,
                y_eval=y_eval,
                model=default_model_training_specs['model'],
                dataset=self.dataset,
                protected_feature=parsed_bias_command['kwargs']['protected'],
                target_feature=self.dataset.label,
                desired_outcome=parsed_bias_command['kwargs']['desired_outcome'],
                lr=float(lr),
                batch_size=int(batch_size),
                num_epochs=int(num_epochs)
            )
        
        else:
            fm = parsed_bias_command['function_name']
            raise NotImplementedError(f'Bias measure {fm} is not implemented.')

        return bias_score

    def _user_custom_compiler(self, syn_data, parsed_user_custom_command):
        """
        Takes a parsed custom user command that involves the calling of a function that was defined by the user in the python scope 
        of the original classifier and passed to this object. It executes the user function to calculate the resulting regularizer
        that is added on the synthesizer loss.

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
    def prepare_data(data, predict_on_features, label, dataset, device=None):
        """
        Static method to prepare the input data for training or prediction by extracting the columns to predict 
        on and converting the label from one-hot encoding to ordinal encoding.

        :param data: (torch.tensor) The input data in one-hot encoding.
        :param predict_on_features: (list) The name of all features to predict on.
        :param label: (str) The name of the target feature.
        :param dataset: (BaseDataset) Instantiated BaseDataset containing all necessary information about the dataset being synthesized.
        :param device: (str) Name of the device on which all torch tensors are to be found.
        :return: (tuple(torch.tensor)) X-tensor containing the columns to precit on, and y-tensor containing the labels in ordinal encoding.
        """
        if device is None:
            device = data.device

        predict_on_features_idx = list(chain.from_iterable(list(idx) for feature_name, idx in dataset.full_one_hot_index_map.items() if feature_name in predict_on_features))
        
        label_idx = dataset.full_one_hot_index_map[label]

        X = data[:, predict_on_features_idx]
        y = (data[:, label_idx] @ torch.arange(len(label_idx), device=device).float().T).long()

        return X, y

    @staticmethod
    def get_default_model_training_specs(input_dim, output_dim, device):
        """
        Static method that prepares the default setup for downstram modelling tasks.

        :param input_dim: (int) Number of input dimension to the model.
        :param output_dim: (int) Number of output dimension of the model.
        :param device: (str) The name of the device on which the model shall be placed
        :return: (dict) Dictionary of default model and training parameters.
        """
        specs = {}
        specs['model'] = LogReg(input_dim, output_dim).to(device)
        specs['lr'] = 0.1
        specs['batch_size'] = 512
        specs['num_epochs'] = 10

        return specs
