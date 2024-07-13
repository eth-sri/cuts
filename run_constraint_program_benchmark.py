import os
import pickle
import argparse
import numpy as np
import torch
from customizable_synthesizer import CuTS
from itertools import product
from utils import evaluate_sampled_dataset, statistics, Timer
from constraints import ConstraintEvaluator
from query import get_all_marginals, query_marginal
from tabular_datasets import ADULT, HealthHeritage, Compas, German
import copy


def main(args):

    # set the random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    device = 'cuda'

    available_datasets = {
        'ADULT': ADULT,
        'HealthHeritage': HealthHeritage,
        'HealthHeritageBinaryAge': HealthHeritage,
        'Compas': Compas,
        'CompasBinaryRace': Compas,
        'German': German
    }
    # instantiate the dataset
    if args.dataset == 'ADULT':
        dataset = available_datasets[args.dataset](drop_education_num=True, device=device)
    elif args.dataset == 'CompasBinaryRace':
        dataset = available_datasets[args.dataset](binary_race=True, device=device)
    elif args.dataset == 'HealthHeritageBinaryAge':
        dataset = available_datasets[args.dataset](binary_age=True, device=device)
    else:
        dataset = available_datasets[args.dataset](device=device)
    full_one_hot = dataset.get_Dtrain_full_one_hot(return_torch=True)
    full_one_hot_test = dataset.get_Dtest_full_one_hot(return_torch=True)

    # prepare everything workload related
    translated_workload = {
        'all_two': 2,
        'all_three': 3,
        'all_three_with_labels': 'all_three_with_labels'
    }
    workload_marginal_names = {
        'all_two': get_all_marginals(list(dataset.features.keys()), 2, downward_closure=False),
        'all_three': get_all_marginals(list(dataset.features.keys()), 3, downward_closure=False)
    }
    workload_marginal_names['all_three_with_labels'] = [m for m in workload_marginal_names['all_three'] if dataset.label in m]
    measured_workload = {m: query_marginal(full_one_hot, m, dataset.full_one_hot_index_map, normalize=True, input_torch=True, max_slice=1000) for m in workload_marginal_names[args.workload]}

    # differential privacy incl. or no
    dp = args.epsilon > 0
    additional_string = '_dp' if dp else ''

    if not dp:
        if args.dataset == 'ADULT':
            programs_and_params = {
                f'eliminate_predictability_sex.txt': {'arguments': {'param1': [0.00133]}, 'denoiser_config': {'finetuning_epochs': 200}},
                f'fairness_downstream_sex.txt': {'arguments': {'param1': [0.0009]}, 'denoiser_config': {'finetuning_epochs': 200}},
                f'minimize_correlation_sex.txt': {'arguments': {'param1': [0.7525]}, 'denoiser_config': {'finetuning_epochs': 200}},
                f'avg_age_30{additional_string}.txt': {'arguments': {'param1': [0.000025]}, 'denoiser_config': {'finetuning_epochs': 200}},
                f'avg_male_female_age{additional_string}.txt': {'arguments': {'param1': [0.0000125]}, 'denoiser_config': {'finetuning_epochs': 200}},
                f'implication2{additional_string}.txt': {'arguments': {'param1': [0.0000075]}, 'denoiser_config': {'finetuning_epochs': 500}},
                f'implication4{additional_string}.txt': {'arguments': {'param1': [0.0000075]}, 'denoiser_config': {'finetuning_epochs': 500}},
                f'implication5{additional_string}.txt': {'arguments': {'param1': [0.0000075]}, 'denoiser_config': {'finetuning_epochs': 500}},
                f'line_constraint1{additional_string}.txt': {'arguments': {'param1': [0.0000025]}, 'denoiser_config': {'finetuning_epochs': 500}},
                f'line_constraint2{additional_string}.txt': {'arguments': {'param1': [0.0000075]}, 'denoiser_config': {'finetuning_epochs': 500}},
            }
        elif args.dataset == 'HealthHeritage':
            programs_and_params = {
                f'implication1.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication2.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication3.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint1.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint2.txt': {'arguments': {'param1': [0.1]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical2.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical3.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
            }
        elif args.dataset == 'HealthHeritageBinaryAge':
            programs_and_params = {
                f'fairness_downstream.txt': {'arguments': {'param1': [0.0005, 0.001, 0.0025, 0.005, 0.0075, 0.01, 0.05, 0.1]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication1.txt': {'arguments': {'param1': [0.001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication2.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication3.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint1.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint2.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical2.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical3.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
            }
        elif args.dataset == 'German':
            programs_and_params = {
                f'fairness_downstream.txt': {'arguments': {'param1': [0.0001, 0.0005, 0.001, 0.005, 0.01]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication1.txt': {'arguments': {'param1': [0.001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication2.txt': {'arguments': {'param1': [0.5]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication3.txt': {'arguments': {'param1': [0.001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint1.txt': {'arguments': {'param1': [0.001, 0.0005]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint2.txt': {'arguments': {'param1': [1.0, 1.5, 2.0]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical1.txt': {'arguments': {'param1': [0.0005, 0.00025, 0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical2.txt': {'arguments': {'param1': [0.5, 0.1, 0.05, 0.01]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
            }
        elif args.dataset == 'CompasBinaryRace':
            programs_and_params = {
                f'fairness_downstream.txt': {'arguments': {'param1': [0.001, 0.0025, 0.005, 0.0075, 0.01, 0.05, 0.1]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication1.txt': {'arguments': {'param1': [0.000001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication2.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication3.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint1.txt': {'arguments': {'param1': [0.01]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint2.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical1.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical2.txt': {'arguments': {'param1': [0.1]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
            }
        elif args.dataset == 'Compas':
            programs_and_params = {
                f'implication1.txt': {'arguments': {'param1': [0.000001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication2.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication3.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint1.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint2.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical1.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical2.txt': {'arguments': {'param1': [0.5]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
            }

    else:
        if args.dataset == 'ADULT':
            programs_and_params = {
                f'fairness_downstream_sex_dp.txt': {'arguments': {'param1': [0.0007]}},
                f'implication2{additional_string}.txt': {'arguments': {'param1': [0.00005]}, 'denoiser_config': {'finetuning_epochs': 200}},
                f'implication4{additional_string}.txt': {'arguments': {'param1': [0.0000125]}, 'denoiser_config': {'finetuning_epochs': 200}},
                f'implication5{additional_string}.txt': {'arguments': {'param1': [0.000375]}, 'denoiser_config': {'finetuning_epochs': 200}},
                f'line_constraint1{additional_string}.txt': {'arguments': {'param1': [0.0000375]}, 'denoiser_config': {'finetuning_epochs': 200}},
                f'line_constraint2{additional_string}.txt': {'arguments': {'param1': [0.0000125]}, 'denoiser_config': {'finetuning_epochs': 200}},
                'fairness_EoO_downstream_sex_desired_dp.txt': {'arguments': {'param1': [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]}, 'denoiser_config': {'finetuning_epochs': 200}},
                'fairness_EO_downstream_sex_desired_dp.txt': {'arguments': {'param1': [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0]}, 'denoiser_config': {'finetuning_epochs': 200}}
            }
        elif args.dataset == 'HealthHeritage':
            programs_and_params = {
                f'implication1_dp.txt': {'arguments': {'param1': [0.1]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication2_dp.txt': {'arguments': {'param1': [0.001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication3_dp.txt': {'arguments': {'param1': [0.1]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint1_dp.txt': {'arguments': {'param1': [1.0]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint2_dp.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical2_dp.txt': {'arguments': {'param1': [0.01]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical3_dp.txt': {'arguments': {'param1': [0.01]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
            }
        elif args.dataset == 'HealthHeritageBinaryAge':
            programs_and_params = {
                f'fairness_downstream_dp.txt': {'arguments': {'param1': [0.0001, 0.001, 0.01, 0.1, 1.0, 5.0]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication1_dp.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication2_dp.txt': {'arguments': {'param1': [0.001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication3_dp.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint1_dp.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint2_dp.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical2_dp.txt': {'arguments': {'param1': [0.001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical3_dp.txt': {'arguments': {'param1': [1.0]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
            }
        elif args.dataset == 'German':
            programs_and_params = {
                f'implication1_dp.txt': {'arguments': {'param1': [0.01]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication2_dp.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication3_dp.txt': {'arguments': {'param1': [0.00005]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint1_dp.txt': {'arguments': {'param1': [0.00005]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint2_dp.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical1_dp.txt': {'arguments': {'param1': [0.001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                # f'statistical2_dp.txt': {'arguments': {'param1': [0.01]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
            }
        elif args.dataset == 'Compas':
            programs_and_params = {
                f'implication1_dp.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication2_dp.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication3_dp.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint1_dp.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint2_dp.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical1_dp.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical2_dp.txt': {'arguments': {'param1': [0.1]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
            }
        elif args.dataset == 'CompasBinaryRace':
            programs_and_params = {
                f'fairness_downstream_dp.txt': {'arguments': {'param1': [0.0001, 0.001, 0.01, 0.1, 1.0]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication1_dp.txt': {'arguments': {'param1': [0.0001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication2_dp.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'implication3_dp.txt': {'arguments': {'param1': [1.0]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint1_dp.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'line_constraint2_dp.txt': {'arguments': {'param1': [0.00001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical1_dp.txt': {'arguments': {'param1': [0.001]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
                f'statistical2_dp.txt': {'arguments': {'param1': [0.1]}, 'denoiser_config': {'finetuning_epochs': 200, 'finetuning_batch_size': 15000}},
            }

    base_path = f'experiment_data/constraint_program_experiments/{args.dataset}/'
    base_path += 'dp_constraints/' if dp else 'non_dp_constraints/'
    eval_base_path = base_path + 'testing_results/'
    if len(args.classifiers) > 1 or args.classifiers[0] != 'xgb':
        eval_base_path += '_'.join([c for c in args.classifiers]) + '/'
    os.makedirs(eval_base_path, exist_ok=True)

    for program_name, program_setups in programs_and_params.items():
        
        stripped_program_name = program_name.split('.')[0]
        if args.baseline_mode:
            eval_save_path = eval_base_path + f'{stripped_program_name}_{args.workload}_{args.n_samples}_{args.n_resamples}_{args.random_seed}_{args.epsilon}_baselines.npy'
        else:    
            eval_save_path = eval_base_path + f'{stripped_program_name}_{args.workload}_{args.n_samples}_{args.n_resamples}_{args.random_seed}_{args.epsilon}.npy'

        print(f'Evaluating {stripped_program_name}, Baseline mode: {args.baseline_mode}')

        if os.path.isfile(eval_save_path) and not args.force:
            print('This experiment has been conducted already, abort')
            continue
        
        else:

            load_path = base_path + f'training_constraints/{program_name}'
            with open(load_path, 'r') as f:
                program = f.read()
                print(program)

            eval_load_path = base_path + f'evaluation_constraints/{stripped_program_name}_eval.txt'
            with open(eval_load_path, 'r') as f:
                eval_program = f.read()

            if dp:
                program = program.replace('<epsilon>', str(args.epsilon))

            if '<epsilon>' in eval_program:
                eval_program = eval_program.replace('<epsilon>', str(args.epsilon))

            param_combinations = list(product(*list(program_setups['arguments'].values())))
            if len(param_combinations) < 2 and args.baseline_mode: # compatibility hack
                param_combinations = param_combinations + param_combinations
            timer = Timer(2) if args.baseline_mode else Timer(len(param_combinations) * args.n_samples)

            collected_data = None
            for i, params_combination in enumerate(param_combinations):

                current_arguments = {arg_name: param for arg_name, param in zip(list(program_setups['arguments'].keys()), params_combination)}

                for sample in range(args.n_samples):

                    if (sample > 0 or i > 1) and args.baseline_mode:
                        continue

                    timer.start()
                    if args.baseline_mode:
                        print(f'Baseline datasets: {i+1}/2    {timer}', end='\r')
                    else:
                        print(f'Parameter Combination: {i+1}/{len(param_combinations)}    Sample: {sample+1}/{args.n_samples}    {timer}', end='\n')

                    denoiser_config = {'finetuning_epochs': 0} if args.baseline_mode else program_setups['denoiser_config']

                    synthesizer = CuTS(
                        constraint_program=program, 
                        workload=translated_workload[args.workload], 
                        random_seed=args.random_seed, 
                        device=device,
                        denoiser_config=denoiser_config
                    )

                    synthesizer.fit(program_arguments=current_arguments, verbose=False)

                    for resample in range(args.n_resamples):
                        
                        if args.baseline_mode and i == 0:
                            synthetic_data = full_one_hot.clone().detach()
                        else:
                            if (program_name.startswith('implication') or program_name.startswith('line_constraint') or program_name.startswith('male_wife')) and not args.baseline_mode:
                                synthetic_data = synthesizer.generate_data_with_rejection_sampling(len(synthesizer.base_data), eval_program)
                            else:
                                synthetic_data = synthesizer.generate_data(len(synthesizer.base_data)).detach()
                        
                        stats = evaluate_sampled_dataset(
                            synthetic_dataset=synthetic_data.detach().clone(),
                            workload=workload_marginal_names[args.workload],
                            true_measured_workload=measured_workload,
                            dataset=synthesizer.dataset,
                            max_slice=1000,
                            random_seed=args.random_seed,
                            classifiers_to_use=args.classifiers
                        )

                        # evaluate the constraints
                        ce = ConstraintEvaluator(
                            program=copy.copy(eval_program),
                            dataset=dataset,
                            base_data=full_one_hot_test.detach().clone(),
                            random_state=args.random_seed,
                            classifiers=args.classifiers,
                            program_arguments=None,
                            device=device
                        )
                        constraint_eval_data = ce.evaluate_constraints(synthetic_data.detach().clone())
                        constraint_stats = []
                        for ced in constraint_eval_data:
                            scores = ced['score']
                            if isinstance(scores, list):
                                constraint_stats += [statistics([score]) for score in scores]
                            else:
                                constraint_stats += [statistics([scores])]

                        if collected_data is None:
                            if args.baseline_mode:
                                collected_data = np.zeros((2, args.n_samples, args.n_resamples, len(stats) + len(constraint_stats), 5))
                            else:
                                collected_data = np.zeros((len(param_combinations), args.n_samples, args.n_resamples, len(stats) + len(constraint_stats), 5))
                        
                        for idx, stat in enumerate(stats):
                            collected_data[i, sample, resample, idx] = stat

                        for l, c_stats in enumerate(constraint_stats):
                            collected_data[i, sample, resample, len(stats)+l] = c_stats

                        # always save
                        np.save(eval_save_path, collected_data)

                    timer.end()
            
            timer.duration()
            np.save(eval_save_path, collected_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('validation_param_search_parser')
    parser.add_argument('--dataset', type=str, default='ADULT', help='Dataset name')
    parser.add_argument('--n_samples', type=int, default=5, help='Number of reruns')
    parser.add_argument('--n_resamples', type=int, default=5, help='Number of resamples')
    parser.add_argument('--random_seed', type=int, default=42, help='Set the random seed')
    parser.add_argument('--workload', type=str, default='all_three_with_labels', help='Set the base workload')
    parser.add_argument('--epsilon', type=float, default=0.0, help='Epsilon for DP constraints')
    parser.add_argument('--baseline_mode', action='store_true', help='Evaluate only single baseline points')
    parser.add_argument('--classifiers', type=str, nargs='+', default=['xgb'], help='List of the evaluation classifiers')
    parser.add_argument('--force', action='store_true', help='Force the execution')
    in_args = parser.parse_args()
    main(in_args)
