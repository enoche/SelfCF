# coding: utf-8
# @Time   : 2021/3/30
# @Author : Xin Zhou
# @Email  : enoche.chow@gmail.com

# UPDATE:

"""
Main entry
##########################
"""


import os
#os.environ['NUMEXPR_MAX_THREADS'] = '64'
#os.environ['NUMEXPR_NUM_THREADS'] = '48'
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import argparse
from utils.quick_start import quick_start


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', type=str, default='SELFCFHE_BPR', help='name of models')
    parser.add_argument('--dataset', '-d', type=str, default='ml-10k', help='name of datasets')

    config_dict = {
        'gpu_id': 0,
    }

    args, _ = parser.parse_known_args()

    quick_start(model=args.model, dataset=args.dataset, config_dict=config_dict, save_model=True)

