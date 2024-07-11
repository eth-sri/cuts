import pyparsing as pp
import re
import copy


class FunctionGrammar:

    """
    PEG grammar describing the parsing of functions in a python-like snytax. The syntax has to follow
    the following pattern:
        my_function(first_argument=value1, second_argument=value2, ..., last_argument=valueN),
    which is then parsed to seperate the function name and the argument groups.
    """
    
    def __init__(self):
        
        # grammar defintion in PEG
        number = pp.Regex(r"[+-]?\d+(:?\.\d*)?(:?[eE][+-]?\d+)?")
        variable = pp.Word(pp.alphanums + '._') | number
        square_bracket_open = pp.Literal('[').suppress()
        square_bracket_closed = pp.Literal(']').suppress()
        bracket_open = pp.Literal('(').suppress()
        bracket_closed = pp.Literal(')').suppress()
        comma = pp.Literal(',').suppress()
        equal = pp.Literal('=').suppress()
        list_content = pp.infixNotation(variable, [(comma, 2, pp.opAssoc.LEFT)]) #| variable
        full_list = pp.Group(square_bracket_open + list_content + square_bracket_closed)
        argument_value = variable | full_list
        argument_pair = pp.Group(variable + equal + argument_value)
        arguments_content = pp.infixNotation(argument_pair, [(comma, 2, pp.opAssoc.LEFT)]) #| argument_pair
        function = pp.Group(variable + bracket_open + arguments_content + bracket_closed)
        
        self.parser = function
    
    def parse(self, string):
        return self.parser.parseString(string).asList()
    

class FirstOrderLogicGrammar:

    """
    PEG grammar to parse first order logical expressions containing comparisons, conjuctions, and disjunctions.
    The expression is parsed into a non-binary operation tree.
    """
    
    def __init__(self):
        
        # grammar definiton in PEG
        number = pp.Regex(r"[+-]?\d+(:?\.\d*)?(:?[eE][+-]?\d+)?")
        variable = pp.Word(pp.alphanums + '_') | number
        operator = pp.Regex(">=|<=|!=|>|<|==").setName("operator")
        comparison = pp.Group(variable + operator + variable)
        logical_expression = pp.infixNotation(comparison, [('AND', 2, pp.opAssoc.LEFT), ('OR', 2, pp.opAssoc.LEFT)])
        
        self.parser = logical_expression
        
    def parse(self, string):
        return self.parser.parseString(string).asList()
    

class StatisticalExpressionGrammar:

    """
    PEG grammar to parse complex statistical expressions with operations within and over conditional expectations, variances, and entropies.
    An example of a parseable complex expression is:
        EXP[3*(custom_function(x)-5)|y==2 OR z in {1, 2, 3}] - VAR[x].
    The expression is parsed into a non-binary operation tree.
    """
    
    def __init__(self):
        
        # grammar defintion in PEG
        operator = pp.Regex(">=|<=|!=|>|<|==").setName("operator")
        scientific = pp.Regex(r"[+-]?\d+(\.\d*)?[eE][+-]?\d+")
        opening_bracket = pp.Literal("[").suppress()
        closing_bracket = pp.Literal("]").suppress()
        separator = pp.Literal("|").suppress()
        variable = scientific | pp.Word(pp.alphas + pp.nums + '_' + ',' + '.' + '{' + '}')
        stat_symbol = pp.Word(pp.alphas)
        arithmetic_expression = pp.infixNotation(variable, [(pp.oneOf("* /"), 2, pp.opAssoc.LEFT, ), (pp.oneOf("+ -"), 2, pp.opAssoc.LEFT, )]) #| variable
        comparison = pp.Group(variable + operator + variable)
        logical_expression = pp.infixNotation(comparison, [("AND", 2, pp.opAssoc.LEFT, ), ("OR", 2, pp.opAssoc.LEFT, )]) | variable

        conditional_statop = pp.Group(stat_symbol + opening_bracket + arithmetic_expression + separator + logical_expression + closing_bracket) | variable
        conditional_expr = pp.infixNotation(conditional_statop, [(pp.oneOf("* /"), 2, pp.opAssoc.LEFT), 
                                                                 (pp.oneOf("+ -"), 2, pp.opAssoc.LEFT), 
                                                                 (pp.oneOf(">= <= != > < =="), 2, pp.opAssoc.LEFT),
                                                                 ("AND", 2, pp.opAssoc.LEFT),
                                                                 ("OR", 2, pp.opAssoc.LEFT)])
        
        self.parser = conditional_expr
    
    def parse(self, string):
        return self.parser.parseString(string).asList()


class ConstraintProgramParser:

    """
    The main parsing class that is able to parse a constraint program line-by-line into operation trees and spec dictionaries 
    that can be converted to pyTorch to enforce the required constraints during training.
    """

    escapes = {
        ' ': 'XXSPACEXX',
        '-': 'XXDASHXX',
        '>=': 'XXGEQXX',
        '<=': 'XXLEQXX',
        '>': 'XXGEXX',
        '<': 'XXLEXX',
        '!=': 'XXNEQXX',
        '==': 'XXEQXX'
    }

    def __init__(self, features=None):
        
        self.function_grammar = FunctionGrammar()
        self.first_order_logic_grammar = FirstOrderLogicGrammar()
        self.statistical_expression_grammar = StatisticalExpressionGrammar()
        self.features = features

    def parse_constraint_program(self, constraint_program, program_arguments=None):
        """
        Takes a prompt adhering to the syntax of the synthesizer constraint language and 
        returns the parsed command, where each row is parsed into an operation tree.
        
        :param constraint_program: (str) The input command with correct syntax.
        :param program_arguments: (dict) Optional arguments to the program.
        :return: (list) For each line an operation tree or spec dictionary.
        """
        constraint_program = ConstraintProgramParser.handle_arguments(constraint_program, program_arguments)
        constraint_program = ConstraintProgramParser.insert_escapes_features(constraint_program, self.features)
        constraint_program = re.sub(r'#.*?\n', '\n', constraint_program) # remove comments
        constraint_program = constraint_program.replace('\n', '')
        parsed_constraint_program = []
        
        parsers = {
            'row constraint': self._row_constraint_parser,
            'implication': self._implication_parser,
            'bias': self._bias_parser,
            'statistical': self._statistical_parser,
            'utility': self._utility_parser,
            'differential privacy': self._differential_privacy_parser,
            'user custom': self._user_custom_parser 
        }
        
        tokenized_constraint_program = self.tokenize_prompt(constraint_program)
        _, dataset_name = tokenized_constraint_program.pop(0)
        
        for line in tokenized_constraint_program:
            if line[0].lower().startswith('end'):
                break
            if len(line) == 3:
                action, command_type, command = line
                param = None
            else:
                action, command_type, param, command = line
                param = float(param[len('param '):])
            original_command = ConstraintProgramParser.remove_escapes(copy.copy(command)).strip()
            command = self.expand_sets(command, self.features)
            parsed_command = parsers[command_type.lower()](command)
            parsed_constraint_program.append({'command_type': command_type.lower(), 
                                              'action': action.lower(), 
                                              'param': param, 
                                              'parsed_command': parsed_command,
                                              'original_command': original_command
                                              })
        return parsed_constraint_program

    @staticmethod
    def handle_arguments(prompt, arguments):
        """
        Handles the arguments to the constraint program that are to be provided in a dictionary. The dictionary keys are the names
        of the arguments, as placed in the constraint program <argument_name>, and the corresponding values are the inputs to which
        the arguments are to be changed.

        :param prompt: (str) The constraint program.
        :param arguments: (dict) The dictionary containing the input arguments.
        :return: (str) The constraint program with the arguments replaced with the desired input values.
        """
        if arguments is None:
            return prompt
        else:
            for argument_name, argument_value in arguments.items():
                prompt = prompt.replace('<' + argument_name + '>', str(argument_value))
            return prompt
    
    @staticmethod
    def tokenize_prompt(prompt):
        """
        A static method that takes a constraint program line and slices at each colon, plus cleans the leading element.

        :param prompt: (str) A string to be tokenized at each colon.
        :return: (list) List of cleaned and tokenized input string tokens.
        """
        line_split_prompt = prompt.split(';')
        tokenized_prompt = []
        for line in line_split_prompt:
            split = re.split(': |:', line)
            split[0] = ''.join([s for s in split[0] if s not in '- '])
            tokenized_prompt.append(split)
        return tokenized_prompt
    
    @staticmethod
    def insert_escapes_features(prompt, features):
        """
        Takes a program and wraps each feature into escape characters for eased parsing. Deals with the special characters
        of a space and dash in feature names and feature values.

        :param prompt: (str) The CuTS program.
        :param features: (dict) The dataset features dictionary.
        :return: (str) The prompt with escaped features.
        """
        possibly_to_escape = []
        for feature_name, feature_domain in features.items():
            if feature_name in prompt or (feature_domain is not None and any([str(ft) in prompt for ft in feature_domain])):
                possibly_to_escape.extend(
                    [feature_name] + [str(ft) for ft in feature_domain if str(ft) in prompt] if feature_domain is not None else []
                )
        for string in possibly_to_escape:
            prompt = prompt.replace(string, ConstraintProgramParser.add_escapes(string))
        
        return prompt
    
    @staticmethod
    def remove_escapes_from_parsed(parsed_expression):
        """
        Recursive method to take a parsed expression tree and replace each string with the non-escaped version.

        :param parsed_expression: (list) The parsed expression containing escaped strings.
        :return: (list) The same parsed expression, but with clean strings.
        """
        cleaned_parsed_expression = []
        for pe in parsed_expression:
            if isinstance(pe, str):
                cleaned_parsed_expression.append(ConstraintProgramParser.remove_escapes(pe))
            else:
                cleaned_parsed_expression.append(ConstraintProgramParser.remove_escapes_from_parsed(pe))
        return cleaned_parsed_expression
    
    @staticmethod
    def add_escapes(string):
        """
        Takes a string and adds the escapes.

        :param string: (str) The string to add the escaped to.
        :return: (str) The escaped string.
        """
        for orig, escaped in ConstraintProgramParser.escapes.items():
            string = string.replace(orig, escaped)
        return string

    @staticmethod
    def remove_escapes(string):
        """
        Takes a string and removes the escapes.

        :param string: (str) The string containing escapes.
        :return: (str) Escape-free string.
        """
        for orig, escaped in ConstraintProgramParser.escapes.items():
            string = string.replace(escaped, orig)
        return string
    
    @staticmethod
    def expand_set_exclusion(expression, features):
        """
        Takes an expression containing not in set expressions, and expands them to chained OR expressions, for instance
        let x indicate a score from 1 to 5 and we have the following expression:
            x not in {1, 2, 3} --> x == 4 OR x == 5.
        This allows us to later use a much simpler parser.

        :param expression: (str) A mathematical/logical expression.
        :param features: (dict) A dictionary that contains the feature as its key and the feature's domain as its value
            to allow us to find the complement of the set.
        :return: (str) The expanded expression.
        """
        pattern = r"(\w+)\s+not in\s+\{([\w\s,-]+)\}"
        match = re.search(pattern, expression)
        while match:
            feature = ConstraintProgramParser.remove_escapes(match.group(1))
            exclude = match.group(2).split(",")
            exclude = [f.strip() for f in exclude]
            full_set = [f for f in features[feature]]
            possibilities = [f for f in full_set if f not in exclude]
            first_possibility = possibilities.pop(0)
            rewritten = f'{feature} == {first_possibility}'
            rewritten = '(' + rewritten + ''.join([f' OR {feature} == {p}' for p in possibilities]) + ')'
            expression = expression[:match.start()] + rewritten + expression[match.end():]
            match = re.search(pattern, expression)
        return ConstraintProgramParser.insert_escapes_features(expression, features)

    @staticmethod
    def expand_set_inclusion(expression):
        """
        Takes an expression containing in set expression, and expands them to chained OR expressions, for instance:
            x in {1, 2, 3} --> x == 1 OR x == 2 OR x == 3.
        This allows us to later use a much simpler parser.

        :param expression: (str) A mathematical/logical expression.
        :return: (str) The expanded expression.
        """
        pattern = r"(\w+)\s+in\s+\{([\w\s,-]+)\}"
        match = re.search(pattern, expression)
        while match:
            feature = match.group(1)
            possibilities = match.group(2).split(",")
            possibilities = [p.strip() for p in possibilities]
            first_possibility = possibilities.pop(0)
            rewritten = f'{feature} == {first_possibility}'
            rewritten = '(' + rewritten + ''.join([f' OR {feature} == {p}' for p in possibilities]) + ')'
            expression = expression[:match.start()] + rewritten + expression[match.end():]
            match = re.search(pattern, expression)
        return expression

    @staticmethod
    def expand_sets(expression, features):
        """
        Takes an expression containing in or not in set expressions and expands them to chained OR expressions.

        :param expression: (str) A mathematical/logic expression.
        :param features: (dict) A dictionary that contains the feature as its key and the feature's domain as its value
            to allow us to find the complement of the set.
        :return: (str) The expanded expression.
        """
        expression = ConstraintProgramParser.expand_set_exclusion(expression, features)
        expression = ConstraintProgramParser.expand_set_inclusion(expression)
        return expression

    @staticmethod
    def adjust_statistical_expression(expression):
        """
        Takes a statistical expression and modifies some elements to be parsable by the PEG grammars contained in the object. The two main
        modifications are:
            - brackets that symbolize function calls are replaced with curled brackets,
            - and non-conditional expressions are extended with a None-condition: EXP[x] --> EXP[x|None].
        These modifications allow us to use simpler parsers.

        :param expression: (str) Statistical expression to be modified.
        :return: (str) The adjusted statistical expression.
        """
        adjusted_expression = ''
        adjustment = '|None'
        opened = False
        separator = False
        index_of_bracket_open = None
        bracket_open = False
        arithmetic_operation = False
        
        expression = expression.replace(', ', ',')
        
        idx = 0
        for s in expression:
            if s == '[':
                opened = True
                adjusted_expression += s
            elif s == '|':
                separator = True
                adjusted_expression += s
            elif s == ']' and opened:
                if not separator:
                    adjusted_expression += adjustment
                    idx += len(adjustment)
                opened = False
                separator = False
                adjusted_expression += s
            elif s == '(' and opened and not separator:
                arithmetic_operation = False
                index_of_bracket_open = idx
                adjusted_expression += s
            elif s == ')' and opened and not separator:
                if not arithmetic_operation:
                    adjusted_expression += '}'
                    adjusted_expression = adjusted_expression[:index_of_bracket_open] + '{' + adjusted_expression[index_of_bracket_open+1:]
                    arithmetic_operation = False
                else:
                    adjusted_expression += s
            else:
                arithmetic_operation = True if s in ['+', '-', '*', '/'] else arithmetic_operation
                adjusted_expression += s
            idx += 1
        
        return adjusted_expression
    
    @staticmethod
    def negate_parsed_logical_expression(parsed_expression):
        """
        Takes a parsed first order logical expression and returnd the parsed negation of it.

        :param parsed_expression: (list) The parsed expression already in a list format to be negated.
        :return: (list) The negated parsed expression.
        """
        negated_expr = []
        for token in parsed_expression:
            if isinstance(token, list):
                negated_expr.append(ConstraintProgramParser.negate_parsed_logical_expression(token))
            elif token == 'AND':
                negated_expr.append('OR')
            elif token == 'OR':
                negated_expr.append('AND')
            elif token in ['>', '<=', '>=', '<', '==', '!=']:
                negated_expr.append(ConstraintProgramParser.negate_operator(token))
            else:
                negated_expr.append(token)
            
        return negated_expr
    
    @staticmethod
    def negate_operator(operator):
        """
        Takes a logical operator from the following options: ['>', '<=', '>=', '<', '==', '!='], and returns its 
        negated counterpart.

        :param operator: (str) One of the following logical operators: ['>', '<=', '>=', '<', '==', '!='].
        :return: (str) The negated operator.
        """
        if operator == ">":
            return "<="
        elif operator == "<=":
            return ">"
        elif operator == ">=":
            return '<'
        elif operator == '<':
            return '>='
        elif operator == '==':
            return '!='
        elif operator == '!=':
            return '=='

    @staticmethod
    def unparse_expression(expression):
        """
        Static method to be used to unparse an arithmetic expression.

        :param expression: (list) Parsed arithmetic expression.
        :return: (str) The unparsed arithmetic expression as a string.
        """
        string = '('
        for token in expression:
            if isinstance(token, str):
                string += token
            else:
                string += ConstraintProgramParser.unparse_expression(token)
        string += ')'
        return string
    
    @staticmethod
    def parsed_expression_to_lambda_function(expression, features):
        """
        Takes a parsed arithmetic expression and turns it into a lambda function, plus it returns the list of
        arguments (features) that are involved in the function.

        :param expression: (list) The parsed expression to be converted.
        :param features: (dict) The dataset feature dictionary.
        :return: (callable, list) The extracted lambda function and a list of all features involved in the function.
        """
        unparsed_expression = ConstraintProgramParser.unparse_expression(expression).replace('}', ')').replace('{', '(')
        features_involved = [feature for feature in features.keys() if feature in unparsed_expression]
        features_involved_return = copy.deepcopy(features_involved)
        lambda_function_arguments = features_involved.pop(0)
        for feature in features_involved:
            lambda_function_arguments += f', {feature}'
        lambda_function = ConstraintProgramParser.insert_escapes_features(f'lambda {lambda_function_arguments}: {unparsed_expression}', features)
        return eval(lambda_function), features_involved_return

    @staticmethod
    def is_leaf_of_first_order_logic_expression(command):
        operators = ['<', '>', '<=', '>=', '==', '!=']
        is_leaf = len(command) == 3 and command[1] in operators
        return is_leaf

    @staticmethod
    def binarize_first_order_logic_operation_tree(command):
        # unpack more
        if isinstance(command, list) and len(command) == 1:
            return ConstraintProgramParser.binarize_first_order_logic_operation_tree(command[0])

        # return leaf, lowest point of recursion
        if ConstraintProgramParser.is_leaf_of_first_order_logic_expression(command) or isinstance(command, str):
            return command
        
        # new tree
        binary_tree = [command[0]]

        # recursively binarize the children
        left_child = ConstraintProgramParser.binarize_first_order_logic_operation_tree(command[1])
        right_child = ConstraintProgramParser.binarize_first_order_logic_operation_tree(command[2:])

        binary_tree.append(left_child)

        # handle potentially more than one right children, this is the key part
        if len(right_child) > 1:
            binary_tree.append(ConstraintProgramParser.binarize_first_order_logic_operation_tree(right_child))
        else:
            binary_tree.append(right_child[0])

        return [binary_tree]

    def _row_constraint_parser(self, command):
        """
        Takes a logical expression applying over a line of data and parses it into a non-binary operation tree.

        :param command: (str) The logical expression to be parsed.
        :return: (list) The resulting operation tree after parsing.
        """
        return ConstraintProgramParser.remove_escapes_from_parsed(self.first_order_logic_grammar.parse(command))

    def _bias_parser(self, command):
        """
        Takes a command containing a python-like function call on a bias measure and parses it into a list containing the function name 
        and a dictionary of keyword arguments.

        :param command: (str) The python-like function call command.
        :return: (list) A list containing the function name and a dictionary of keyword arguments.
        """
        return self._function_parser(command)

    def _statistical_parser(self, command):
        """
        Takes a command of statistical expressions and parses it into an operation tree.

        :param command: (str) The statistical expression command that is to be parsed.
        :return: (list) The resulting operation tree.
        """
        adjusted_command = self.adjust_statistical_expression(command)
        return ConstraintProgramParser.remove_escapes_from_parsed(self.statistical_expression_grammar.parse(adjusted_command))

    def _utility_parser(self, command):
        """
        Takes a utility function call in python-like syntax and returns the function named and argument-value pairs parsed into a 
        list and a dictionary.

        :param command: The python-like function call command.
        :return: (list) A list containing the function name and a dictionary of keyword arguments.
        """
        return self._function_parser(command)

    def _implication_parser(self, command):
        """
        Takes an implication of two first order logical expressions: A IMPLIES B. First it separates the string at the implication sign
        and then parses each side with a first order logic parser. Finally, it binarizes the operation tree of the antecedent and it 
        negates the consequent such that we can later target all violation of the implication in the data.

        :param command: (str) An implication A IMPLIES B.
        :return: (dict) A dictionary of the parsed antecedent and consequent negated.
        """
        split_command = command.split('IMPLIES')
        parsed_command = {}
        parsed_command['antecedent'] = (
            [ConstraintProgramParser.binarize_first_order_logic_operation_tree(ConstraintProgramParser.remove_escapes_from_parsed(self.first_order_logic_grammar.parse(split_command[0])))]
        )
        parsed_command['neg_consequent'] = (
            self.negate_parsed_logical_expression(ConstraintProgramParser.remove_escapes_from_parsed(self.first_order_logic_grammar.parse(split_command[1])))
        )
        return parsed_command

    def _function_parser(self, command):
        """
        Takes a command containing a python-like function call and parses it into a list containing the function name 
        and a dictionary of keyword arguments.

        :param command:(str) The python-like function call command.
        :return: (list) A list containing the function name and a dictionary of keyword arguments.
        """
        parsed_string_list = ConstraintProgramParser.remove_escapes_from_parsed(self.function_grammar.parse(command)[0])
        function_name = parsed_string_list.pop(0).lower()
        # the parser does not fold single arguments --> not the most elegant solution
        parsed_string_list = parsed_string_list[0] if isinstance(parsed_string_list[0][0], list) else parsed_string_list
        parsed_arguments = {}
        for arg_pairs in parsed_string_list:
            parsed_arguments[arg_pairs[0]] = arg_pairs[1][0] if isinstance(arg_pairs[1], list) else arg_pairs[1]
        parsed_string = {'function_name': function_name, 'kwargs': parsed_arguments}
        return parsed_string

    def _differential_privacy_parser(self, command):
        """
        Takes a command describing a differential privacy requirement separated by an and. For example:
            epsilon = 1.0 and delta = 1e-9.
        It parses then this into a dictionary:
            {'epsilon': 1.0, 'delta': 1e-9}.

        :param command: (str) The differential privacy comment as specified above.
        :return: (dict) The parsed dictionary as specified above.
        """
        command = command.lower().replace(' ', '')
        parsed_command = {}
        split_command = command.split(',')
        for spec in split_command:
            option, value = spec.split('=')
            parsed_command[option] = float(value)
        return parsed_command

    def _user_custom_parser(self, command):
        """
        Takes a command containing a python-like function call and parses it into a list containing the function name 
        and a dictionary of keyword arguments.

        :param command:(str) The python-like function call command.
        :return: (list) A list containing the function name and a dictionary of keyword arguments.
        """
        return self._function_parser(command)
