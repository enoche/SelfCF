# coding: utf-8
# UPDATE:

"""
Main entry
##########################
"""


import os
import argparse
from utils.quick_start import quick_start
os.environ['NUMEXPR_MAX_THREADS'] = '48'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFHE_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')

    config_dict = {
        'gpu_id': 0,
        'epochs': 2,
        'n_layers': [4],
        'reg_weight': [0.01],
        'momentum': [0.05],
        'dropout': [0.1],
    }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)


