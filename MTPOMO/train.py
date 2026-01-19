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

from VRPTrainer import VRPTrainer


##########################################################################################
# parameters
# problem_type:
# If problem_type = 'unified': trained on 16.6% MDVRP, 16.6% MDOVRP, 16.6% MDVRPB, 16.6% MDVRPTW, 16.6% MDVRPL, 16.6% MDVRPI
# If problem_type = 'full_task_set': trained on all MDVRP variants
# problem_type can also be MDVRP, MDOVRP, MDVRPB, MDVRPTW, MDVRPL, MDVRPI and their any combinations, e.g., MDOVRPBLTW
# curriculum:
# If True: perform a curriculum schedule of variants through training
# If False: sample all variants uniformly since the beginning of training
# curriculum_schedule:
# schedule to introduce two-, three- and four-constraint variants during training 
# for instance [0.3, 0.6, 0.9] means that when 30% of epochs are finished, variants with two constraints are 
# introduced, at 60% variants with three constraints, and 90% variants with four constraints

env_params = {
    'problem_type': 'unified',
    'problem_size': 50,
    'pomo_size': 52,
    'depot_size': 3,
    'curriculum': True,
    'curriculum_schedule': [0.3, 0.6, 0.9]
}

model_params = {
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128**(1/2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'eval_type': 'argmax',
    'norm': 'instance',
    'use_film': False,
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
        'weight_decay': 1e-6
    },
}

trainer_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'epochs': 300,
    'train_episodes': 100000,
    'train_batch_size': 128,
    'use_autocast': True,
    'seed': 1234,
    'config_name': 'train_'+env_params['problem_type']+'_n'+str(env_params['problem_size'])+'_mtpomo_film_curriculum',
    'logging': {
        'model_save_interval': 20,
        'validation_interval': 5,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_unified_100.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': './result/20251023_012447_train_full_task_set_n50_mtpomo_film_curriculum',  # directory path of pre-trained model and log files saved.
        'epoch': 260 # epoch version of pre-trained model to laod.
    }
}

logger_params = {
    'log_file': {
        'desc': 'train_'+env_params['problem_type']+'_n'+str(env_params['problem_size'])+'_mtpomo_film_curriculum',
        'filename': 'run_log'
    }
}


##########################################################################################
# main

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    log_path = create_logger(**logger_params)
    _print_config()

    seed = trainer_params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)
    
    print(">> Start VRP Training.")
    trainer = VRPTrainer(env_params=env_params,
                  model_params=model_params,
                  optimizer_params=optimizer_params,
                  trainer_params=trainer_params,
                  log_path=log_path)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():
    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]



##########################################################################################

if __name__ == "__main__":
    main()
