# coding: utf-8
# @Time   : 2021/3/30
# @Author : Xin Zhou
# @Email  : enoche.chow@gmail.com

# UPDATE:

"""
Main entry
##########################
"""

import argparse
from utils.quick_start import quick_start


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFED_LGN', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-100k', help='name of datasets')

    config_dict = {
        #'stopping_step': 2,
        'epochs': 2
    }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=False)


