import torch
import os
from tabular_datasets import ADULT, German, HealthHeritage, Compas
from denoiser import Denoiser
from constraints import ConstraintProgramParser, ConstraintEvaluator, ConstraintCompiler
import copy
import pickle
from query import get_all_marginals
import numpy as np
from utils import create_kfold_index_splits, evaluate_sampled_dataset, statistics, Timer
from itertools import product


class CuTS:

    def __init__(self, constraint_program, workload=None, denoiser_config=None, user_custom_functions=None, 
                 path=None, random_seed=42, device='cuda'):

        self.device = device
        self.random_seed = random_seed
        self.workload = 'all_three_with_labels' if workload is None else workload
        self.constraint_program = constraint_program
        self.denoiser_config = {} if denoiser_config is None else denoiser_config
        self.user_custom_functions = user_custom_functions

        # available tabular_datasets to synthesize
        available_datasets = {
            'adult': ADULT,
            'german': German,
            'healthheritage': HealthHeritage,
            'healthheritagebinaryage': HealthHeritage,
            'compas': Compas,
            'compasbinaryrace': Compas,
        }

        # extract the name of the dataset from the prompt and instantiate the dataset
        _, dataset_name = ConstraintProgramParser.tokenize_prompt(constraint_program)[0]
        if dataset_name.lower() == 'compasbinaryrace':
            self.dataset = available_datasets[dataset_name.lower()](binary_race=True, device=self.device)
        elif dataset_name.lower() == 'healthheritagebinaryage':
            self.dataset = available_datasets[dataset_name.lower()](binary_age=True, device=self.device)
        else:
            self.dataset = available_datasets[dataset_name.lower()](device=self.device)

        self.path = f'experiment_data/programmable_synthesizer_experiments/{dataset_name.lower()}/trained_models' if path is None else path

        # get the marginal combinations in the workload
        workload_dim = 3 if self.workload == 'all_three_with_labels' else self.workload
        self.workload_marginal_names = get_all_marginals(list(self.dataset.features.keys()), workload_dim, downward_closure=False)
        if self.workload == 'all_three_with_labels':
            self.workload_marginal_names = [m for m in self.workload_marginal_names if self.dataset.label in m]

        # will be filled later, once calling fit
        self.base_data = None
        self.parser = None
        self.compiler = None
        self.base_model = None
        self.measured_workload = None
        self.epsilon, self.delta = np.inf, 1e-9

        # will be filled new every time fit is called
        self.finetuned_model = None
        self.non_dp_constraints_present = False
    
    def kfold_cross_validation(self, program_argument_domains, k=5, resamples=5, verbose=False, max_slice=1000, force=False, save=False, single_sample=False):
        """
        Conducts k-fold cross validation on the argument domains for the constraint params.

        :param program_argument_domains: (dict) Pass the argument domains to search.
        :param verbose: (bool) Toggle for verbose training of the underlying models.
        :param resamples: (int) Number of times we resample the dataset to understand the effect of the data generation noise.
        :param max_slice: (int) Max slice for marginal querying.
        :param force: (bool) Force the fitting.
        :param save: (bool) Save the fitted base model. Not recommended within this method.
        :param single_sample: (bool) If True, then instead of doing k splits, we only evaluate on the first split.
        :return: (dict) Dictionary containing the mean performance over all parameter combinations.
        """
        # create a dummy argument setup for the initial parsing -- these have no influence on the base model
        dummy_program_arguments = {arg_name: arg_vals[0] for arg_name, arg_vals in program_argument_domains.items()}

        self._prepare_base_model_and_marginals(
            program_arguments=dummy_program_arguments,
            verbose=verbose,
            max_slice=max_slice,
            force=force,
            save=save
        )

        # create splits through indexing
        index_splits = create_kfold_index_splits(self.base_data.size()[0], k)

        k_fold_eval_data = {}

        # iterate over the splits and the parameter spaces
        param_combinations = list(product(*list(program_argument_domains.values())))
        timer = Timer(len(param_combinations) * k)
        for i, params_combination in enumerate(param_combinations):

            current_arguments = {arg_name: param for arg_name, param in zip(list(program_argument_domains.keys()), params_combination)}

            collected_data_for_param_comb = None

            for j, (train_idx, eval_idx) in enumerate(index_splits):

                if single_sample and j > 0:
                    break

                timer.start()
                if verbose:
                    if collected_data_for_param_comb is not None:
                        print(f'Param combination: {i+1}/{len(param_combinations)}    Split: {j+1}/{k}    XGB acc: {100*np.mean(collected_data_for_param_comb[j, :, 3, 0]):.1f}%    Constraint Score: {np.mean(collected_data_for_param_comb[j, :, 6, 0]):.2f}    {timer}', end='\n')
                    else:
                        print(f'Param combination: {i+1}/{len(param_combinations)}    Split: {j+1}/{k}    {timer}', end='\n')
                    
                # ------------- Finetune ------------- #
                # instantiate the compiler
                self.compiler = ConstraintCompiler(
                    program=self.constraint_program,
                    dataset=self.dataset,
                    base_data=self.base_data[train_idx].detach().clone(),  # here is the trick
                    program_arguments=current_arguments,
                    user_custom_functions=self.user_custom_functions,
                    device=self.device
                )
                # finetune
                finetuned_model = copy.deepcopy(self.base_model)
                finetuned_model.fit(
                    algorithm='input',
                    target_marginals=self.measured_workload,
                    n_epochs=self.denoiser_config['finetuning_epochs'],
                    batch_size=self.denoiser_config['finetuning_batch_size'],
                    subsample=self.denoiser_config['finetuning_subsampling'],
                    loss_to_use=self.denoiser_config['finetuning_loss'],
                    max_slice=max_slice,
                    constraint_compiler=self.compiler,
                    verbose=False
                )
                
                for n in range(resamples):
                    # ------------- Evaluate ------------- #
                    # sample a dataset
                    synthetic_data = finetuned_model.generate_data(len(train_idx), sample=(finetuned_model.head == 'softmax')).detach().clone()
                    
                    # first do a global evaluation
                    tv_stats, l2_stats, js_stats, acc_stats, bac_stats, f1_stats = evaluate_sampled_dataset(
                        synthetic_dataset=synthetic_data.detach().clone(),
                        workload=list(self.measured_workload.keys()),
                        true_measured_workload=self.measured_workload,
                        dataset=self.dataset,
                        max_slice=1000,
                        random_seed=self.random_seed
                    )

                    # evaluator
                    ce = ConstraintEvaluator(
                        program=self.constraint_program,
                        dataset=self.dataset,
                        base_data=self.base_data[eval_idx].detach().clone(),
                        xgb_random_state=self.random_seed,
                        program_arguments=current_arguments,
                        device=self.device
                    )

                    constraint_eval_data = ce.evaluate_constraints(synthetic_data.detach().clone())
                    constraint_stats = []
                    for single_constraint_stats in constraint_eval_data:
                        # keep only the XGB balanced accuracy for utility experiments here, so we only report a single stat 
                        score = single_constraint_stats['score'][1] if isinstance(single_constraint_stats['score'], list) else single_constraint_stats['score']
                        constraint_stats.append(score)
                    
                    if collected_data_for_param_comb is None:
                        if single_sample:
                            collected_data_for_param_comb = np.zeros((1, resamples, 6 + len(constraint_stats)))
                        else:
                            collected_data_for_param_comb = np.zeros((k, resamples, 6 + len(constraint_stats)))
                    
                    collected_data_for_param_comb[j, n, 0] = tv_stats[0]
                    collected_data_for_param_comb[j, n, 1] = l2_stats[0]
                    collected_data_for_param_comb[j, n, 2] = js_stats[0]
                    collected_data_for_param_comb[j, n, 3] = acc_stats[0]
                    collected_data_for_param_comb[j, n, 4] = bac_stats[0]
                    collected_data_for_param_comb[j, n, 5] = f1_stats[0]

                    for l, c_stats in enumerate(constraint_stats):
                        collected_data_for_param_comb[j, n, 6+l] = c_stats

                timer.end()
            k_fold_eval_data[params_combination] = (np.nanmean(collected_data_for_param_comb, axis=(0, 1)), np.nanstd(collected_data_for_param_comb, axis=(0, 1)))
         
        return k_fold_eval_data

    def fit(self, program_arguments=None, verbose=False, max_slice=1000, force=False, finetune=True, save=True):
        """
        Fits the synthesizer according to the prompt. Note that at this point it is possible to pass arguments to the
        program.

        :param program_arguments: (dict) Optional, pass arguments to the constraint program.
        :param verbose: (bool) Toggle for verbose training of the underlying models.
        :param max_slice: (int) Max slice for marginal querying.
        :param force: (bool) Force the fitting.
        :param finetune: (bool) If set to false then even though constraints are given, there is no fine-tuning conducted.
        :param save: (bool) Save the fitted base model.
        :return: self
        """
        # take care of the base model, train it if necessary, load if it exists
        self._prepare_base_model_and_marginals(
            program_arguments=program_arguments,
            verbose=verbose,
            max_slice=max_slice,
            force=force,
            save=save
        )

        # now we can instantiate the compiler
        self.compiler = ConstraintCompiler(
            program=self.constraint_program,
            dataset=self.dataset,
            base_data=self.base_data.detach().clone(),
            program_arguments=program_arguments,
            user_custom_functions=self.user_custom_functions,
            device=self.device
        )

        # finetune if there is any command to finetune on
        self.finetuned_model = copy.deepcopy(self.base_model)
        if self.non_dp_constraints_present and finetune:
            self.finetuned_model.fit(
                algorithm='input',
                target_marginals=self.measured_workload,
                n_epochs=self.denoiser_config['finetuning_epochs'],
                batch_size=self.denoiser_config['finetuning_batch_size'],
                subsample=self.denoiser_config['finetuning_subsampling'],
                loss_to_use=self.denoiser_config['finetuning_loss'],
                max_slice=max_slice,
                constraint_compiler=self.compiler,
                verbose=verbose
            )

        return self
    
    def _prepare_base_model_and_marginals(self, program_arguments=None, verbose=False, max_slice=1000, force=False, save=True):
        """
        Private method to prepare and train the base model and the measured marginals for finetuning. It saves the model it
        trains at the current configuration, and in case this configuration has already been trained, then it loads it.

        :param program_arguments: (dict) Optional, pass arguments to the constraint program.
        :param verbose: (bool) Toggle for verbose training of the underlying models.
        :param max_slice: (int) Max slice for marginal querying.
        :param force: (bool) Force the fitting.
        :param save: (bool) Save the fitted base model.
        :return: self
        """
        # start by parsing the program to find out if we have DP
        self.parser = ConstraintProgramParser(
            features=self.dataset.features
        )
        parsed_program = self.parser.parse_constraint_program(self.constraint_program, program_arguments)

        dp_command = [command['parsed_command'] for command in parsed_program if command['command_type'] == 'differential privacy']
        is_dp = len(dp_command) > 0

        self.non_dp_constraints_present = len([command for command in parsed_program if command['command_type'] != 'differential privacy']) > 0

        if is_dp:
            self.epsilon, self.delta = dp_command[0]['epsilon'], dp_command[0]['delta']
            
        # extract data from the dataset
        train_full_one_hot = self.dataset.get_Dtrain_full_one_hot(return_torch=True)
        self.n_dimensions = train_full_one_hot.size()[1]
        
        # get the config --> extend all missing values with defaults
        default_denoiser_config = self._get_default_denoiser_config(dp=is_dp)
        for key, val in default_denoiser_config.items():
            if key in self.denoiser_config:
                self.denoiser_config[key] = self.denoiser_config[key]
            else:
                self.denoiser_config[key] = val
        
        # get the paths
        base_model_path, workload_path = self._build_paths()
        
        # check if the model and the workloads exist, if no, train again
        if os.path.isfile(base_model_path) and os.path.isfile(workload_path) and not force:
            with open(base_model_path, 'rb') as f:
                self.base_model = pickle.load(f)
            with open(workload_path, 'rb') as f:
                self.measured_workload = pickle.load(f)
            
            # put everything on the correct device
            self.base_model.to(self.device)
            for measured_marginal in self.measured_workload.values():
                measured_marginal.to(self.device)
            
        else:

            # instantiate the model
            self.base_model = Denoiser(
                num_features=train_full_one_hot.size()[1],
                one_hot_index_map=self.dataset.full_one_hot_index_map,
                in_size=100,
                layout=self.denoiser_config['architecture_layout'],
                architecture_type=self.denoiser_config['architecture_type'],
                head=self.denoiser_config['head'],
                noise_type='gaussian',
                device=self.device
            )
            train_algorithm = 'aim' if is_dp else 'input'
            # train the model
            self.measured_workload = self.base_model.fit(
                epsilon=self.epsilon,
                delta=self.delta,
                algorithm=train_algorithm,
                full_one_hot_dataset=train_full_one_hot,
                workload=self.workload_marginal_names,
                T=None,
                alpha=0.9,
                anneal='adaptive',
                horizontal=self.denoiser_config['base_model_horizontal'],
                keep_running_average=True,
                data_len=None,
                target_marginals=None,
                n_epochs=self.denoiser_config['base_model_epochs'],
                batch_size=self.denoiser_config['base_model_batch_size'],
                subsample=self.denoiser_config['base_model_subsampling'],
                loss_to_use=self.denoiser_config['base_model_loss'],
                max_slice=max_slice,
                return_measurements=True,
                verbose=verbose
            )

            # save everything
            if save:
                with open(base_model_path, 'wb') as f:
                    pickle.dump(self.base_model, f)
                with open(workload_path, 'wb') as f:
                    pickle.dump(self.measured_workload, f)

        # extract the base data
        if is_dp:
            self.base_data = self.base_model.generate_data(self.base_model.data_len, sample=(self.base_model.head == 'softmax')).detach().clone()
        else:
            self.base_data = train_full_one_hot
        
        return self

    def generate_data(self, size, base_generator=False):
        """
        Generates synthetic data, either from the finetuned generator or from the base one.

        :param size: (int) The number of rows to generate.
        :return: (torch.tensor) The generated data.
        """
        if base_generator:
            return self.base_model.generate_data(size, sample=(self.base_model.head == 'softmax')).detach().clone()
        else:
            return self.finetuned_model.generate_data(size, sample=(self.finetuned_model.head == 'softmax')).detach().clone()
        

    def generate_data_with_rejection_sampling(self, size, rejection_program, max_trials=1000, program_arguments=None, base_generator=False, verbose=False):
        """
        A simple implementation for rejection sampling to ensure that ROW CONSTRAINT and IMPLICATION type constraint are satisfied on the
        generated data. Note that this can be both used to make an only pre-trained CuTS naively satisfy these constraints, or it can 
        also be used to make a fine-tuned CuTS be even better at ensuring that every condition is satisfied.

        :param size: (int) The desired size of the generated dataset.
        :param rejection_program: (str) The program containing the constraint to be enforced through the rejection sampling.
        :param max_trials: (int) The maximum number of trials until we give up resampling.
        :param program_arguments: (dict) Optionally, arguments to the program. Note that this dictionary has to be given if the program
            contains place-holder for the parameters in the program.
        :param base_generator: (bool) Toggle to use only the pre-trained generator.
        :param verbose: (bool) Toggle verbosity for sampling.
        :return: (torch.tensor) The rejection sampled "clean" data.
        """
        rejection_compiler = ConstraintCompiler(
            program=rejection_program,
            dataset=self.dataset,
            base_data=None,
            program_arguments=program_arguments,
            user_custom_functions=None,
            device=self.device,
        )            

        current_size = size
        data_rejection_sampled = None

        for trial in range(max_trials):

            if current_size == 0:
                break

            if verbose:
                print(f'Trial: {trial+1}    Rows to resample: {current_size}                                 ', end='\n')

            # generate data
            syn_data = self.generate_data(size=current_size, base_generator=base_generator)

            violating_rows = torch.zeros(current_size).to(self.device)
            for constraint in rejection_compiler.parsed_program:

                # ignore any non logical constraints
                if constraint['command_type'] == 'row constraint':
                    
                    violating_rows += rejection_compiler._recursive_row_constraint_compiler(syn_data, ConstraintProgramParser.negate_parsed_logical_expression(constraint['parsed_command']))
                
                elif constraint['command_type'] == 'implication':
                    
                    antecedent_row_mask = rejection_compiler._recursive_row_constraint_compiler(syn_data, constraint['parsed_command']['antecedent'], compensate_redundancy=True)
                    neg_consequent_row_mask = rejection_compiler._recursive_row_constraint_compiler(syn_data, constraint['parsed_command']['neg_consequent'])
                    violating_rows += antecedent_row_mask * neg_consequent_row_mask
                
                else:
                    continue

            # get the indices where the rows are violating and delete these rows from the data
            accepted_idx = np.argwhere(violating_rows.cpu().numpy() < 0.1).flatten()  # should be binary, but this should cover in case we have some small deviations
            accepted_data = syn_data[accepted_idx]
            current_size = current_size - len(accepted_idx)
            current_size = 2 if current_size == 1 else current_size  # we always need at least 2 datapoints for BN
                
            if data_rejection_sampled is None:
                data_rejection_sampled = accepted_data
            else:
                data_rejection_sampled = torch.cat([data_rejection_sampled, accepted_data.view(-1, data_rejection_sampled.size()[1])], axis=0)

        return data_rejection_sampled[:size]

    def _get_default_denoiser_config(self, dp=False):
        """
        Returns the default configurations for either DP or non-DP training.

        :param dp: (bool) Toggle to show if the training is DP.
        :return: (dict) Configs.
        """

        # non dp base params
        denoiser_config_non_dp = {
            'architecture_type': 'residual',
            'architecture_layout': [100, 200, 200, 200, self.n_dimensions],
            'head': 'gumbel',#'hard_softmax',
            'base_model_epochs': 2000,
            'base_model_loss': 'total_variation',
            'base_model_batch_size': 15000,
            'base_model_subsampling': 16,
            'base_model_horizontal': True,
            'finetuning_epochs': 200,#40,
            'finetuning_loss': 'total_variation',
            'finetuning_batch_size': 15000,
            'finetuning_subsampling': None
        }

        # dp base params
        denoiser_config_dp = {
            'architecture_type': 'residual',
            'architecture_layout': [100, 200, 200, 200, self.n_dimensions],
            'head': 'gumbel',
            'base_model_epochs': 1000,
            'base_model_loss': 'total_variation',
            'base_model_batch_size': 1000,
            'base_model_subsampling': None,
            'base_model_horizontal': True,
            'finetuning_epochs': 200,
            'finetuning_loss': 'total_variation',
            'finetuning_batch_size': 15000,
            'finetuning_subsampling': None
        }

        if dp:
            return denoiser_config_dp
        else:
            return denoiser_config_non_dp

    def _build_paths(self, k=None, s=None):
        """
        Builds the paths for the models and the workloads to be stored at.

        :param k: (int) K of the k-fold cross validation.
        :param s: (int) The current fold of the K.
        :return: (tuple of str) The path for the base model and the path for the reference training workloads.
        """
        architecture_type = self.denoiser_config['architecture_type']
        architecture_layout = str(self.denoiser_config['architecture_layout']).replace(' ', '')
        head = self.denoiser_config['head']
        base_model_epochs = self.denoiser_config['base_model_epochs']
        base_model_loss = self.denoiser_config['base_model_loss']
        base_model_batch_size = self.denoiser_config['base_model_batch_size']
        base_model_subsampling = self.denoiser_config['base_model_subsampling']
    
        # base model
        base_model_folder = f'{self.path}/base_models'
        base_model_path = f'{base_model_folder}/base_model_{self.dataset.name.lower()}'
        base_model_path += f'_{self.workload}'
        base_model_path += f'_{architecture_type}'
        base_model_path += f'_{architecture_layout}'
        base_model_path += f'_{head}'
        base_model_path += f'_{base_model_epochs}'
        base_model_path += f'_{base_model_loss}'
        base_model_path += f'_{base_model_batch_size}'
        base_model_path += f'_{base_model_subsampling}'
        base_model_path += f'_{self.random_seed}'
        if self.epsilon is not None:
            base_model_path += f'_{self.epsilon}'
        if k is not None:
            base_model_path += f'_fold_{s}_{k}'
        base_model_path += '.pickle'

        # workload
        workload_folder = f'{self.path}/workloads'
        workload_path = f'{workload_folder}/workloads_{self.workload}_{self.random_seed}'
        if self.epsilon is not None:
            workload_path += f'_{self.epsilon}'
        if k is not None:
            base_model_path += f'_fold_{s}_{k}'
        workload_path += '.pickle'

        # make the directories path
        os.makedirs(base_model_folder, exist_ok=True)
        os.makedirs(workload_folder, exist_ok=True)

        return base_model_path, workload_path
