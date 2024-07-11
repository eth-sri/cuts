import numpy as np
import torch
from tabular_datasets import HealthHeritage, Compas, German
from argparse import ArgumentParser


def main(args):

    datasets = {
        'Health_Heritage': HealthHeritage,
        'Compas': Compas,
        'German': German
    }

    # set the random seed
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)

    if args.dataset == 'Compas':
        dataset = datasets[args.dataset](random_state=args.random_seed, train_test_ratio=args.split_ratio, binary_race=args.binary_fairness, split_from_file=False)
        file_postfix = f'{args.split_ratio}_{args.random_seed}_{args.binary_fairness}.npy'
    elif args.dataset == 'Health_Heritage':
        dataset = datasets[args.dataset](random_state=args.random_seed, train_test_ratio=args.split_ratio, binary_age=args.binary_fairness, split_from_file=False)
        file_postfix = f'{args.split_ratio}_{args.random_seed}_{args.binary_fairness}.npy'
    else:
        dataset = datasets[args.dataset](random_state=args.random_seed, train_test_ratio=args.split_ratio, split_from_file=False)
        file_postfix = f'{args.split_ratio}_{args.random_seed}.npy'

    Xtrain, ytrain = dataset.get_Xtrain().cpu().numpy(), dataset.get_ytrain().cpu().numpy()
    Xtest, ytest = dataset.get_Xtest().cpu().numpy(), dataset.get_ytest().cpu().numpy()

    name_data = {
        'xtrain': Xtrain, 
        'ytrain': ytrain, 
        'xtest': Xtest, 
        'ytest': ytest
    }

    for data_name, data in name_data.items():
        np.save(f'tabular_datasets/{args.dataset}/presplit_{data_name}_{file_postfix}', data)


if __name__ == '__main__':
    parser = ArgumentParser('split_parser')
    parser.add_argument('--dataset', type=str, help='Pass the dataset to split')
    parser.add_argument('--random_seed', default=42, type=int, help='Set the random seed of the split')
    parser.add_argument('--split_ratio', default=0.2, type=float, help='Set the split ratio')
    parser.add_argument('--binary_fairness', action='store_true', help='Toggle for the fairness binarized version, \
                        only relevant for some datasets')
    in_args = parser.parse_args()
    main(in_args)
