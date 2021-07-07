# coding: utf-8
# @Time   : 2021/04/01
# @Author : Xin Zhou
# @Email  : enoche.chow@gmail.com

# UPDATE:

"""
Run application
##########################
"""
from logging import getLogger
from itertools import product
from utils.dataset import RecDataset
from utils.dataloader import TrainDataLoader, EvalDataLoader
from utils.logger import init_logger
from utils.configurator import Config
from utils.utils import init_seed, get_model, get_trainer


def quick_start(model, dataset, config_dict, save_model=True):
    # merge config dict
    config = Config(model, dataset, config_dict)
    init_logger(config)
    logger = getLogger()
    # print config infor
    logger.info(config)
    # load data
    dataset = RecDataset(config)
    # print dataset statistics
    logger.info(str(dataset))

    train_dataset, valid_dataset, test_dataset = dataset.split(config['split_ratio'])
    logger.info('\n====Training====\n'+str(train_dataset))
    logger.info('\n====Validation====\n'+str(valid_dataset))
    logger.info('\n====Testing====\n'+str(test_dataset))

    # wrap into dataloader
    train_data = TrainDataLoader(config, train_dataset, batch_size=config['train_batch_size'])
    (valid_data, test_data) = (EvalDataLoader(config, valid_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']),
                               EvalDataLoader(config, test_dataset, additional_dataset=train_dataset, batch_size=config['eval_batch_size']))

    ############ Dataset loadded, run model
    hyper_ret = []
    val_metric = config['valid_metric'].lower()
    best_test_value = 0.0
    idx = best_test_idx = 0

    logger.info('\n\n=================================\n\n')

    # hyper-parameters
    hyper_ls = []
    for i in config['hyper_parameters']:
        hyper_ls.append(config[i] or [None])
    # combinations
    combinators = list(product(*hyper_ls))
    total_loops = len(combinators)
    for hyper_tuple in combinators:
        for j, k in zip(config['hyper_parameters'], hyper_tuple):
            config[j] = k

        logger.info('========={}/{}: Parameters:{}={}======='.format(
            idx+1, total_loops, config['hyper_parameters'], hyper_tuple))
        # random seed
        init_seed(config['seed'])
        # model loading and initialization
        model = get_model(config['model'])(config, train_data).to(config['device'])
        if idx==0:
            logger.info(model)
        # trainer loading and initialization
        trainer = get_trainer()(config, model)
        # model training
        best_valid_score, best_valid_result = trainer.fit(train_data, valid_data=valid_data, test_data=test_data, saved=save_model)
        # model evaluation
        test_result = trainer.evaluate(test_data, load_best_model=save_model, is_test=True, idx=idx)
        #########
        hyper_ret.append((hyper_tuple, best_valid_result, test_result))

        # save best test
        if test_result[val_metric] > best_test_value:
            best_test_value = test_result[val_metric]
            best_test_idx = idx
        idx += 1

        logger.info('best valid result: {}'.format(best_valid_result))
        logger.info('test result: {}'.format(test_result))
        logger.info('Current BEST: Parameters: {}={},\n '
                    'best valid: {},\n best test: {}\n\n\n'.format(config['hyper_parameters'],
            hyper_ret[best_test_idx][0], hyper_ret[best_test_idx][1], hyper_ret[best_test_idx][2]))

    # log info
    logger.info('\n============All Over=====================')
    for (p, k, v) in hyper_ret:
        logger.info('Parameters: {}={},\n best valid: {},\n best test: {}'.format(config['hyper_parameters'], p, k, v))

