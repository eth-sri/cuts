import os
import pickle
import argparse
import numpy as np
import torch
import time
from customizable_synthesizer import CuTS
from itertools import product
from utils import evaluate_sampled_dataset, statistics, Timer
from constraints import ConstraintEvaluator
from query import get_all_marginals, query_marginal
from tabular_datasets import ADULT, HealthHeritage, German, Compas
import copy
import re


def main(args):

    # set the random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    device = 'cuda'

    params = {
        'ADULT': {
            1: {'param1': 9e-4, 'param2': 2.5e-5, 'param3': 1.25e-5, 'param4': 7.5e-6, 'param5': 7.5e-6},
            2: {'param1': 7.5e-6, 'param2': 7.5e-6, 'param3': 7.5e-6, 'param4': 2.5e-6, 'param5': 7.5e-6}
        },
        'HealthHeritage': {
            2: {'param1': 1e-4, 'param2': 1e-5, 'param3': 1e-4, 'param4': 1e-5, 'param5': 1e-1}
        },
        'HealthHeritageBinaryAge': {
            1: {'param1': 0.0005, 'param2': 0.0001, 'param3': 0.0001, 'param4': 0.0001, 'param5': 0.0001} # param1: 0.005
        },
        'Compas': {
            2: {'param1': 0.000001, 'param2': 0.0001, 'param3': 0.00001, 'param4': 0.00001, 'param5': 0.00001}
        },
        'CompasBinaryRace': {
            1: {'param1': 0.0075, 'param2': 0.00001, 'param3': 0.1, 'param4': 0.00001, 'param5': 0.00001}
        },
        'German': {
            1: {'param1': 0.0005, 'param2': 0.00025, 'param3': 0.05, 'param4': 0.001, 'param5': 0.5},
            2: {'param1': 0.001, 'param2': 0.5, 'param3': 0.001, 'param4': 0.0005, 'param5': 1.5}
        },
    }

    if args.dataset == 'ADULT':
    
        constraints_to_chain_option_1 = {
            'fairness_downstream_sex.txt': {'param1': 9e-4},
            'fairness_downstream_sex_avg_age_30.txt': {'param1': 9e-4, 'param2': 2.5e-5},
            'fairness_downstream_sex_avg_age_30_avg_male_female_age.txt': {'param1': 9e-4, 'param2': 2.5e-5, 'param3': 1.25e-5},
            'fairness_downstream_sex_avg_age_30_avg_male_female_age_implication5.txt': {'param1': 9e-4, 'param2': 2.5e-5, 'param3': 1.25e-5, 'param4': 7.5e-6},
            'fairness_downstream_sex_avg_age_30_avg_male_female_age_implication5_implication4.txt': {'param1': 9e-4, 'param2': 2.5e-5, 'param3': 1.25e-5, 'param4': 7.5e-6, 'param5': 7.5e-6}
        }

        constraints_to_chain_option_2 = {
            'implication2.txt': {'param1': 7.5e-6},
            'implication_2_implication4.txt': {'param1': 7.5e-6, 'param2': 7.5e-6},
            'implication_2_implication4_implication5.txt': {'param1': 7.5e-6, 'param2': 7.5e-6, 'param3': 7.5e-6},
            'implication_2_implication4_implication5_line_constraint1.txt': {'param1': 7.5e-6, 'param2': 7.5e-6, 'param3': 7.5e-6, 'param4': 2.5e-6},
            'implication_2_implication4_implication5_line_constraint1_line_constraint2.txt': {'param1': 7.5e-6, 'param2': 7.5e-6, 'param3': 7.5e-6, 'param4': 2.5e-6, 'param5': 7.5e-6}
        }

        constraints_to_chain_option_3 = {
            'base_program.txt': {},
            's1.txt': {'param1': 2.5e-5},
            's1_s2.txt': {'param1': 2.5e-5, 'param2': 1.25e-5},
            's1_s2_i3.txt': {'param1': 2.5e-5, 'param2': 1.25e-5, 'param3': 7.5e-6},
            's1_s2_i3_i2.txt': {'param1': 2.5e-5, 'param2': 1.25e-5, 'param3': 7.5e-6, 'param4': 7.5e-6},
            's1_s2_i3_i2_s3.txt': {'param1': 2.5e-5, 'param2': 1.25e-5, 'param3': 7.5e-6, 'param4': 7.5e-6, 'param5': 0.7525},
            's1_s2_i3_i2_s3_r2.txt': {'param1': 2.5e-5, 'param2': 1.25e-5, 'param3': 7.5e-6, 'param4': 7.5e-6, 'param5': 0.7525, 'param6': 0.0000075},
            's1_s2_i3_i2_s3_r2_i1.txt': {'param1': 2.5e-5, 'param2': 1.25e-5, 'param3': 7.5e-6, 'param4': 7.5e-6, 'param5': 0.7525, 'param6': 0.0000075, 'param7': 0.0000075},
            's1_s2_i3_i2_s3_r2_i1_fair.txt': {'param1': 2.5e-5, 'param2': 1.25e-5, 'param3': 7.5e-6, 'param4': 7.5e-6, 'param5': 0.7525, 'param6': 0.0000075, 'param7': 0.0000075, 'param8': 9e-4},
        }
    
    else:

        constraints_to_chain_option_1 = {
            'fairness_downstream.txt': {param_name: param for param_name, param in params[args.dataset][args.option].items() if int(param_name[-1]) < 2},
            'fairness_downstream_statistical1.txt': {param_name: param for param_name, param in params[args.dataset][args.option].items() if int(param_name[-1]) < 3},
            'fairness_downstream_statistical1_statistical2.txt': {param_name: param for param_name, param in params[args.dataset][args.option].items() if int(param_name[-1]) < 4},
            'fairness_downstream_statistical1_statistical2_implication3.txt': {param_name: param for param_name, param in params[args.dataset][args.option].items() if int(param_name[-1]) < 5},
            'fairness_downstream_statistical1_statistical2_implication3_implication2.txt': {param_name: param for param_name, param in params[args.dataset][args.option].items()},
        }

        constraints_to_chain_option_2 = {
            'implication1.txt': {param_name: param for param_name, param in params[args.dataset][args.option].items() if int(param_name[-1]) < 2},
            'implication1_implication2.txt': {param_name: param for param_name, param in params[args.dataset][args.option].items() if int(param_name[-1]) < 3},
            'implication1_implication2_implication3.txt': {param_name: param for param_name, param in params[args.dataset][args.option].items() if int(param_name[-1]) < 4},
            'implication1_implication2_implication3_line_constraint1.txt': {param_name: param for param_name, param in params[args.dataset][args.option].items() if int(param_name[-1]) < 5},
            'implication1_implication2_implication3_line_constraint1_line_constraint2.txt': {param_name: param for param_name, param in params[args.dataset][args.option].items()},
        }


    
    if args.dataset == 'ADULT':
        if args.option == 1:
            eval_program_name = 'fairness_downstream_sex_avg_age_30_avg_male_female_age_implication5_implication4_eval.txt'
            constraints_and_params = constraints_to_chain_option_1
        elif args.option == 2:
            eval_program_name = 'implication_2_implication4_implication5_line_constraint1_line_constraint2_eval.txt'
            constraints_and_params = constraints_to_chain_option_2
        elif args.option == 3:
            eval_program_name = 's1_s2_i3_i2_s3_r2_i1_fair_eval.txt'
            constraints_and_params = constraints_to_chain_option_3
    else:
        eval_program_name = 'fairness_downstream_statistical1_statistical2_implication3_implication2_eval.txt' if args.option == 1 else 'implication1_implication2_implication3_line_constraint1_line_constraint2_eval.txt'
        constraints_and_params = constraints_to_chain_option_1 if args.option == 1 else constraints_to_chain_option_2
    if args.baseline_mode:
        constraints_and_params = {'base_program.txt': None}

    # dataset preps
    datasets = {
        'ADULT': ADULT,
        'German': German,
        'Compas': Compas,
        'CompasBinaryRace': Compas,
        'HealthHeritage': HealthHeritage,
        'HealthHeritageBinaryAge': HealthHeritage
    }
    if args.dataset == 'CompasBinaryRace':
        dataset = datasets[args.dataset](binary_race=True, device=device)
    elif args.dataset == 'HealthHeritageBinaryAge':
        dataset = datasets[args.dataset](binary_age=True, device=device)
    else:
        dataset = datasets[args.dataset](device=device)
    full_one_hot_train = dataset.get_Dtrain_full_one_hot(return_torch=True)
    full_one_hot_test = dataset.get_Dtest_full_one_hot(return_torch=True)

    # workload for eval
    workload_all_three_with_labels = [m for m in get_all_marginals(list(dataset.features.keys()), 3, downward_closure=False) if dataset.label in m]
    measured_workload = {m: query_marginal(full_one_hot_train, m, dataset.full_one_hot_index_map, normalize=True, input_torch=True, max_slice=1000) for m in workload_all_three_with_labels}

    base_load_path_train = f'experiment_data/constraint_ablation/{args.dataset.lower()}_constraint_ablation/option{args.option}/training_constraints/'
    base_load_path_test = f'experiment_data/constraint_ablation/{args.dataset.lower()}_constraint_ablation/option{args.option}/evaluation_constraints/'
    base_save_path = f'experiment_data/constraint_ablation/{args.dataset.lower()}_constraint_ablation/evaluation_results/'
    os.makedirs(base_save_path, exist_ok=True)

    full_save_path = f'{base_save_path}collected_data_option{args.option}_{args.n_samples}_{args.n_resamples}_{args.random_seed}.npy'
    if args.baseline_mode:
        full_save_path = f'{base_save_path}collected_data_option{args.option}_{args.n_samples}_{args.n_resamples}_{args.random_seed}_baseline.npy'
    
    denoiser_config = {'finetuning_epochs': 500}

    num_consts = 8 if args.option == 3 else 5
    collected_data = np.zeros((num_consts, args.n_samples, args.n_resamples, 6 + num_consts, 5))
    timer = Timer(len(constraints_and_params) * args.n_samples)
    for sample in range(args.n_samples):
        for i, (constraint_file, params) in enumerate(constraints_and_params.items()):
            
            timer.start()
            print(f'Sample: {sample+1}/{args.n_samples}    Constraint: {constraint_file}    {timer}                              ', end='\r')
            # load the train and the eval constraints
            with open(f'{base_load_path_train}{constraint_file}', 'r') as f:
                training_program = f.read()
            
            with open(f'{base_load_path_test}{eval_program_name}', 'r') as f:
                eval_program = f.read()
            
            # instantiate CuTS
            synthesizer = CuTS(
                constraint_program=training_program,
                workload='all_three_with_labels',
                random_seed=args.random_seed,
                device=device,
                denoiser_config=denoiser_config
            )

            time_start = time.time()
            synthesizer.fit(program_arguments=params, finetune=True)
            duration = time.time() - time_start
            if args.save_time:
                with open(f'{base_save_path}time_{args.option}.txt', 'a') as f:
                    f.write(str(i) + ': ' + str(duration) + '\n')

            for resample in range(args.n_resamples):
                
                if 'implication' in constraint_file or 'line_constraint' in constraint_file:
                    training_program_param_stripped = re.sub(r'(?i)param\s<[^>\s]*>', 'PARAM 0.1', training_program)
                    synthetic_data = synthesizer.generate_data_with_rejection_sampling(len(synthesizer.base_data), training_program_param_stripped)
                else:
                    synthetic_data = synthesizer.generate_data(len(synthesizer.base_data)).detach()
                
                tv_stats, l2_stats, js_stats, acc_stats, bac_stats, f1_stats = evaluate_sampled_dataset(
                    synthetic_dataset=synthetic_data.detach().clone(),
                    workload=workload_all_three_with_labels,
                    true_measured_workload=measured_workload,
                    dataset=synthesizer.dataset,
                    max_slice=1000,
                    random_seed=args.random_seed
                )

                # evaluate the constraints
                ce = ConstraintEvaluator(
                    program=copy.copy(eval_program),
                    dataset=dataset,
                    base_data=full_one_hot_test.detach().clone(),
                    random_state=args.random_seed,
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
                
                collected_data[i, sample, resample, 0] = tv_stats
                collected_data[i, sample, resample, 1] = l2_stats
                collected_data[i, sample, resample, 2] = js_stats
                collected_data[i, sample, resample, 3] = acc_stats
                collected_data[i, sample, resample, 4] = bac_stats
                collected_data[i, sample, resample, 5] = f1_stats

                for l, c_stats in enumerate(constraint_stats):
                    collected_data[i, sample, resample, 6+l] = c_stats

                # save every time
                np.save(full_save_path, collected_data)

            timer.end()
    np.save(full_save_path, collected_data)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('ablation')
    parser.add_argument('--dataset', type=str, default='ADULT', help='Choose the dataset to run on')
    parser.add_argument('--option', type=int, default=1, help='Choose chaining options')
    parser.add_argument('--n_samples', type=int, default=5, help='Number of reruns')
    parser.add_argument('--n_resamples', type=int, default=5, help='Number of resamples')
    parser.add_argument('--random_seed', type=int, default=42, help='Set the random seed')
    parser.add_argument('--baseline_mode', action='store_true', help='Toggle for baseline mode')
    parser.add_argument('--save_time', action='store_true', help='Toggle for saving time')
    parser.add_argument('--force', action='store_true', help='Force the execution')
    in_args = parser.parse_args()
    main(in_args)
