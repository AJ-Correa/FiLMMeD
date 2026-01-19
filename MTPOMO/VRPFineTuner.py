import random
import torch
import torch.nn.functional as F
from torch import nn
import copy
from logging import getLogger

import os

from MDVRPEnv import MDVRPEnv
from VRPEnv import VRPEnv
from torch.optim import Adam as Optimizer
from torch.optim.lr_scheduler import MultiStepLR as Scheduler
from torch.cuda.amp import autocast, GradScaler

from utils.utils import *


class VRPFineTuner:
    def __init__(self,
                 env_params,
                 model_params,
                 tuner_params,
                 model,
                 test_task=None,
                 log_path=None):

        # save arguments
        self.env_params = env_params
        self.num_depots = self.env_params['depot_size']
        self.model_params = model_params
        self.tuner_params = tuner_params
        self.log_path = log_path

        self.alpha = tuner_params['alpha']

        # result folder, logger
        if self.log_path is not None:
            self.logger = getLogger(name='tuner')
            self.result_folder = get_result_folder()
            self.result_log = LogData()
            self.log_path = log_path

        # cuda
        USE_CUDA = self.tuner_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tuner_params['cuda_device_num']
            device = torch.device('cuda', cuda_device_num)
        else:
            device = torch.device('cpu')
        self.device = device

        # ENV and MODEL
        if self.num_depots > 1:
            self.env = MDVRPEnv(**self.env_params)
        else:
            self.env = VRPEnv(**self.env_params)
        self.model = copy.deepcopy(model)

        optimizer_params = {
            'optimizer': {
                'lr': self.tuner_params['finetuning_lr'],
                'weight_decay': self.tuner_params['finetuning_weight_decay']
            },
            'scheduler': {
                'milestones': self.tuner_params['finetuning_lr_milestones'],
                'gamma': self.tuner_params['finetuning_lr_gamma']}
        }

        self.optimizer = Optimizer(self.model.parameters(), **optimizer_params['optimizer'])
        self.scheduler = Scheduler(self.optimizer, **optimizer_params['scheduler'])
        if self.tuner_params['use_autocast']:
            self.scaler = GradScaler()

        if test_task is not None:
            self.env_params['problem_type'] = test_task
            self.task_set = [test_task]
        else:
            if self.env_params['problem_type'] == 'full_task_set':
                if self.num_depots > 1:
                    self.task_set = ["MDVRP", "MDOVRP", "MDVRPB", "MDVRPL", "MDVRPTW", "MDOVRPTW",
                                     "MDOVRPB", "MDOVRPL", "MDVRPBL", "MDVRPBTW", "MDVRPLTW", "MDOVRPBL",
                                     "MDOVRPBTW", "MDOVRPLTW", "MDVRPBLTW", "MDOVRPBLTW", "MDVRPI", "MDVRPIB",
                                     "MDVRPIL", "MDVRPITW", "MDVRPIBL", "MDVRPIBTW", "MDVRPILTW", "MDVRPIBLTW"]
                else:
                    self.task_set = ["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW",
                                     "OVRPB", "OVRPL", "VRPBL", "VRPBTW", "VRPLTW", "OVRPBL",
                                     "OVRPBTW", "OVRPLTW", "VRPBLTW", "OVRPBLTW"]
            elif self.env_params['problem_type'] == 'unified':
                if self.num_depots > 1:
                    self.task_set = ["MDVRP", "MDOVRP", "MDVRPB", "MDVRPL", "MDVRPTW", "MDVRPI"]
                else:
                    self.task_set = ["CVRP", "OVRP", "VRPB", "VRPL", "VRPTW", "OVRPTW"]
            else:
                self.task_set = [self.env_params['problem_type']]

        # Restore
        self.start_epoch = 1
        model_load = tuner_params.get('model_load', False)
        if model_load:
            if model_load['enable']:
                self.start_epoch = 1 + model_load['epoch']
                self.scheduler.last_epoch = model_load['epoch'] - 1

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):

        self.log_or_print(">> Start VRP Fine-tuning.")
        self.time_estimator.reset(self.start_epoch)
        start = time.time()
        for epoch in range(self.start_epoch, self.tuner_params['finetuning_epochs'] + 1):
            self.log_or_print('=================================================================')

            # Tune
            tuning_score, tuning_loss = self._tune_one_epoch(epoch)
            self.scheduler.step()
            end = time.time()

            if self.log_path is not None:
                excel_log_path = '%s%s_.csv' % (f"./{self.log_path}/", f"{self.tuner_params['config_name']}")
                if epoch == 1:
                    print(f'generate {excel_log_path}')
                    with open(excel_log_path, 'w') as f:
                        f.write('time,epoch,loss,cost\n')

                # save results to a CSV file
                with open(excel_log_path, 'a') as f:
                    f.write('%dmin%dsec,%d,%1.3f,%1.3f\n' % ((end - start) // 60, (end - start) % 60,
                                                             epoch,
                                                             tuning_score,
                                                             tuning_loss))
            start = time.time()

            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.tuner_params[
                'finetuning_epochs'])
            self.log_or_print("Epoch {:3d}/{:3d}({:.2f}%): Elapsed: {} minutes".format(
                epoch, self.tuner_params['finetuning_epochs'], epoch / self.tuner_params['finetuning_epochs'] * 100,
                elapsed_time_str))

            all_done = (epoch == self.tuner_params['finetuning_epochs'])

            # Save Model
            if (self.log_path is not None) and (
                    all_done or (epoch % self.tuner_params['logging']['model_save_interval']) == 0):
                self.log_or_print("Saving fine-tuned model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.log_or_print(" *** Fine-tuning Done *** ")
                # self.logger.info("Now, printing log array...")
                # util_print_log_array(self.logger, self.result_log)

    def _tune_one_epoch(self, epoch):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        tuning_num_episode = self.tuner_params['finetuning_episodes']
        episode = 0

        while episode < tuning_num_episode:
            remaining = tuning_num_episode - episode
            batch_size = min(self.tuner_params['finetuning_batch_size'], remaining)
            if self.env_params['problem_type'] == 'full_task_set' or self.env_params['problem_type'] == 'unified':
                selected = random.choice(self.task_set)
            else:
                selected = self.task_set[0]

            avg_score, avg_loss = self._tune_one_batch(batch_size, selected)
            score_AM.update(avg_score.item(), batch_size)
            loss_AM.update(avg_loss.item(), batch_size)

            episode += batch_size

            self.log_or_print('Epoch {:3d}: Tuning {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}, Variant: {}'
                              .format(epoch, episode, tuning_num_episode, 100. * episode / tuning_num_episode,
                                      score_AM.avg, loss_AM.avg, selected))

        # Print Once, for each epoch
        self.log_or_print('Epoch {:3d}: Tuning ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                          .format(epoch, 100. * episode / tuning_num_episode,
                                  score_AM.avg, loss_AM.avg))
        torch.cuda.empty_cache()

        return score_AM.avg, loss_AM.avg

    def _tune_one_batch(self, batch_size, selected_task):

        env_params = copy.deepcopy(self.env_params)
        env_params['problem_type'] = selected_task

        if self.num_depots > 1:
            env = MDVRPEnv(**env_params)
        else:
            env = VRPEnv(**env_params)

        self.model.train()
        env.load_problems(batch_size)
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)
        prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)
        route_len = torch.zeros(size=(batch_size, env.pomo_size))

        # POMO Rollout
        state, reward, done = env.pre_step()
        while not done:
            if self.tuner_params['use_autocast']:
                with autocast(dtype=torch.float16):  # <── FP16 forward
                    selected, prob = self.model(state)
            else:
                selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)
            route_len = torch.where(selected < (
                self.num_depots if not self.env_params['depot_trick'] else self.env_params['num_virtual_depots']),
                                    state.selected_count, route_len)

        if self.tuner_params['loss_type'] == 'po_loss':
            loss = self.preference_among_pomo_loss_fn(reward, prob_list)
        else:
            loss = self.reinforcement_learning_loss_fn(reward, prob_list)

        # update model
        if self.tuner_params['use_autocast']:
            self.optimizer.zero_grad(set_to_none=True)

            # backward with gradient scaling
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Score
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean()  # negative sign to make positive value
        # print(score_mean)

        del env, reset_state, state, reward, done, prob_list, selected

        return score_mean, loss

    def preference_among_pomo_loss_fn(self, reward, prob):  # PO
        preference = reward[:, :, None] > reward[:, None, :]
        # shape: (batch, pomo, pomo)

        log_prob = self.alpha * torch.log(prob).sum(2)
        log_prob_pair = log_prob[:, :, None] - log_prob[:, None, :]
        pf_log = torch.log(F.sigmoid(log_prob_pair))
        loss = -torch.mean(pf_log * preference)

        return loss

    def reinforcement_learning_loss_fn(self, reward, prob):
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        return loss_mean

    def log_or_print(self, msg):
        if self.log_path is not None:
            self.logger.info(msg)
        else:
            print(msg)

