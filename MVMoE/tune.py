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

from VRPModel import MOEModel as Model
from VRPFineTuner import VRPFineTuner as Tuner

##########################################################################################
# parameters
# problem_type:
# If problem_type = 'full_task_set' and depot_size > 1: solves all 24 MDVRP variants
# Elif problem_type = 'full_task_set' and depot_size == 1: solves all 16 single-depot VRP variants
# problem_type can also be MDVRP, MDOVRP, MDVRPB, MDVRPTW, MDVRPL, MDVRPI and their any combinations, e.g., MDOVRPBLTW
env_params = {
    'problem_type': 'full_task_set',
    'problem_size': 50,
    'pomo_size': 50,
    'depot_size': 1,
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

tuner_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'use_autocast': True,
    'finetuning_epochs': 300,
    'finetuning_episodes': 100000,
    'finetuning_batch_size': 128,
    'loss_type': 'po_loss',  # 'po_loss' or 'rl_loss'
    'finetuning_lr': 3e-4,
    'finetuning_weight_decay': 1e-6,
    'finetuning_lr_milestones': [270, 295],
    'finetuning_lr_gamma': 0.1,
    'alpha': 0.03,  # only used for PO loss
    'seed': 1234,
    'config_name': 'tune_' + env_params['problem_type'] + '_n' + str(
        env_params['problem_size']) + '_mvmoe_film_tune_singledepot',
    'logging': {
        'model_save_interval': 25,
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
        'enable': True,  # enable if loading a checkpoint from an unfinished fine-tuning session
        'path': './result/20251121_043832_tune_full_task_set_n50_mvmoe_film_tune_singledepot',
        # directory path of pre-trained model and log files saved.
        'epoch': 400,  # epoch version of pre-trained model to load
    },
}

logger_params = {
    'log_file': {
        'desc': 'tune_'+env_params['problem_type']+'_n'+str(env_params['problem_size'])+'_mvmoe_film_tune_singledepot',
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

    seed = tuner_params['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    torch.cuda.manual_seed_all(seed)

    # Restore
    if USE_CUDA:
        cuda_device_num = tuner_params['cuda_device_num']
        torch.cuda.set_device(cuda_device_num)
        device = torch.device('cuda', cuda_device_num)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        device = torch.device('cpu')
        torch.set_default_tensor_type('torch.FloatTensor')
    model = Model(**model_params)
    model_load = tuner_params['model_load']
    checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
    checkpoint = torch.load(checkpoint_fullname, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    tuner = Tuner(env_params=env_params,
                  model_params=model_params,
                  tuner_params=tuner_params,
                  model=model,
                  log_path=log_path)

    copy_all_src(tuner.result_folder)

    tuner.run()


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
