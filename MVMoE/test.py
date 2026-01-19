##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config

import os
import sys
import torch
import random
import numpy as np

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from VRPTester import VRPTester as Tester

##########################################################################################
# parameters
# problem_type:
# If problem_type = 'full_task_set' and depot_size > 1: solves all 24 MDVRP variants
# Elif problem_type = 'full_task_set' and depot_size == 1: solves all 16 single-depot VRP variants
# problem_type can also be MDVRP, MDOVRP, MDVRPB, MDVRPTW, MDVRPL, MDVRPI and their any combinations, e.g., MDOVRPBLTW
env_params = {
    'problem_type': 'full_task_set',
    'problem_size': 50,
    'pomo_size': 52,
    'depot_size': 3,
    'depot_trick': False,  # only for single-depot problems, when depot_size = 1
    'num_virtual_depots': 3,
}

model_params = {
    'model_type': 'MOE',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'decoder_layer_num': 1,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'num_experts': 4,
    'eval_type': 'argmax',
    'norm': 'instance',
    'norm_loc': 'norm_last',
    'topk': 2,
    'expert_loc': ['Enc0', 'Enc1', 'Enc2', 'Enc3', 'Enc4', 'Enc5', 'Dec'],
    'routing_level': 'node',
    'routing_method': 'input_choice',
    'use_film': True,
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': './checkpoints/MDVRP50/',
        # directory path of pre-trained model and log files saved.
        'epoch': 300,  # epoch version of pre-trained model to load
    },
    'random_problems': False,
    'num_random_problems': 1000,
    'test_batch_size': 250,
    'augmentation_type': '8d',  # this parameter can be '1', '8', 'd' or '8d'
    'aug_batch_size': 250,
    'enable_finetuning': False,
    'use_autocast': True,
    'seed': 1234,
    'tuner_params': {
        'use_cuda': USE_CUDA,
        'cuda_device_num': CUDA_DEVICE_NUM,
        'use_autocast': True,
        'finetuning_epochs': 1,
        'finetuning_episodes': 10000,
        'finetuning_batch_size': 128,
        'loss_type': 'po_loss',
        'finetuning_lr': 3e-4,
        'finetuning_weight_decay': 1e-6,
        'finetuning_lr_milestones': [270, 295],
        'finetuning_lr_gamma': 0.1,
        'alpha': 0.03,  # only used for PO loss
    }
}
if tester_params['augmentation_type'] != '1':
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    seed = tester_params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(seed)

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    tester.run()


def _set_debug_mode():
    global tester_params
    tester_params['test_episodes'] = 10


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    main()
