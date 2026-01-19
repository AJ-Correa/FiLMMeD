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

# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils


##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src

from VRPTester import VRPTester as Tester


##########################################################################################
env_params = {
    'depot_trick': False,  # only for single-depot instances
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
        'path': './result/MDVRP100/MVMoE - fulltaskset, curriculum, film/20251030_232729_train_full_task_set_n100_mvmoe_film_curriculum',  # directory path of pre-trained model and log files saved.
        'epoch': 300,  # epoch version of pre-trained model to load
    },
    'dataset_path': '../data/MDVRP-LIB/Cordeau-MDVRPTW',
    'augmentation_type': '8d', # this parameter can be '1', '8', 'd' or '8d'
    'use_autocast': True,
    'sample_size': 1,
    'seed': 1234,
    'run_local_search': False,
    'ls_params': {
        'num_ls_iters': 100,
        'granular_neighborhood': 20,
    }
}


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
                    tester_params=tester_params,
                    is_benchmark=True)

    if 'CVRP-LIB' in tester_params['dataset_path']:
        tester._solve_cvrplib_()
    elif 'MDVRP-LIB' in tester_params['dataset_path']:
        tester._solve_mdvrplib_()
    else:
        raise NotImplementedError


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
