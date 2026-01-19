import torch
import random
from logging import getLogger

from MDVRPEnv import MDVRPEnv as Env
from VRPModel import MOEModel as Model

from torch.optim import Adam as Optimizer
from torch.amp import autocast, GradScaler

from utils.utils import *


class VRPTrainer:
    def __init__(self,
                 env_params,
                 model_params,
                 optimizer_params,
                 trainer_params,
                 log_path):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.optimizer_params = optimizer_params
        self.trainer_params = trainer_params

        # result folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()
        self.result_log = LogData()
        self.log_path = log_path

        # cuda
        USE_CUDA = self.trainer_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.trainer_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            self.device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            self.device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')

        # Main Components
        self.model = Model(**self.model_params)
        self.env = Env(**self.env_params)
        self.optimizer = Optimizer(self.model.parameters(), **self.optimizer_params['optimizer'])
        if self.trainer_params['use_autocast']:
            self.scaler = GradScaler(device='cuda' if self.trainer_params['use_cuda'] else 'cpu')

        if self.env_params['curriculum']:
            # self.task_set = ['MDVRP', 'MDOVRP', 'MDVRPB', 'MDVRPL', 'MDVRPTW', 'MDVRPI']
            self.task_set = ['MDVRP', 'MDOVRP', 'MDVRPB', 'MDVRPL', 'MDVRPTW', 'MDVRPI', 'MDVRPBTW', 'MDOVRPTW',
                             'MDVRPITW']
        else:
            if self.env_params['problem_type'] == 'full_task_set':
                self.task_set = ["MDVRP", "MDOVRP", "MDVRPB", "MDVRPL", "MDVRPTW", "MDOVRPTW",
                                 "MDOVRPB", "MDOVRPL", "MDVRPBL", "MDVRPBTW", "MDVRPLTW", "MDOVRPBL",
                                 "MDOVRPBTW", "MDOVRPLTW", "MDVRPBLTW", "MDOVRPBLTW", "MDVRPI", "MDVRPIB",
                                 "MDVRPIL", "MDVRPITW", "MDVRPIBL", "MDVRPIBTW", "MDVRPILTW", "MDVRPIBLTW"]
            elif self.env_params['problem_type'] == 'unified':
                self.task_set = ['MDVRP', 'MDOVRP', 'MDVRPB', 'MDVRPL', 'MDVRPTW', 'MDVRPI']
            else:
                self.task_set = [self.env_params['problem_type']]

        num_tasks = len(self.task_set)
        uniform_prob = 1.0 / num_tasks
        self.task_probs = {task: uniform_prob for task in self.task_set}

        validation_filepath = f"../data/pyvrp_results_vrp{self.env_params['problem_size']}.json"
        with open(validation_filepath, "r") as f:
            self.validation_objectives = json.load(f)

        # Restore
        self.start_epoch = 1
        model_load = trainer_params['model_load']
        if model_load['enable']:
            checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
            checkpoint = torch.load(checkpoint_fullname, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.start_epoch = 1 + model_load['epoch']
            self.result_log.set_raw_data(checkpoint['result_log'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.logger.info('Saved model successfully loaded !!')

            if self.env_params['curriculum']:
                if model_load['epoch'] >= int(self.trainer_params['epochs'] * self.env_params['curriculum_schedule'][0]):
                    #self.task_set.extend(["MDOVRPTW",
                    #            "MDOVRPB", "MDOVRPL", "MDVRPBL", "MDVRPBTW", "MDVRPLTW", "MDVRPIB",
                    #             "MDVRPIL", "MDVRPITW"])
                    self.task_set.extend(["MDOVRPB", "MDOVRPL", "MDVRPBL", "MDVRPLTW", "MDVRPIB",
                                 "MDVRPIL",])
                if model_load['epoch'] >= int(self.trainer_params['epochs'] * self.env_params['curriculum_schedule'][1]):
                    self.task_set.extend(["MDOVRPBL",
                                 "MDOVRPBTW", "MDOVRPLTW", "MDVRPBLTW", "MDVRPIBL", "MDVRPIBTW", "MDVRPILTW"])
                if model_load['epoch'] >= int(self.trainer_params['epochs'] * self.env_params['curriculum_schedule'][2]):
                    self.task_set.extend(["MDOVRPBLTW", "MDVRPIBLTW"])
                num_tasks = len(self.task_set)
                uniform_prob = 1.0 / num_tasks
                self.task_probs = {task: uniform_prob for task in self.task_set}

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):

        self.time_estimator.reset(self.start_epoch)
        start = time.time()
        for epoch in range(self.start_epoch, self.trainer_params['epochs']+1):
            self.logger.info('=================================================================')

            # lr decay (by 10) to speed up convergence at 90th iteration
            if epoch in [int(self.trainer_params['epochs'] * 0.9)]:
                self.optimizer_params['optimizer']['lr'] /= 10
                for group in self.optimizer.param_groups:
                    group["lr"] /= 10
                    print(">> LR decay to {}".format(group["lr"]))

            # Train
            train_score, train_loss = self._train_one_epoch(epoch)
            model_save_interval = self.trainer_params['logging']['model_save_interval']
            validation_interval = self.trainer_params['logging']['validation_interval']

            excel_log_path = '%s%s_.csv' % (f"./{self.log_path}/", f"{self.trainer_params['config_name']}")
            if epoch == 1:
                print(f'generate {excel_log_path}')
                with open(excel_log_path, 'w') as f:
                    f.write('time,epoch,loss,cost\n')

            end = time.time()
            # save results to a CSV file
            with open(excel_log_path, 'a') as f:
                f.write('%dmin%dsec,%d,%1.3f,%1.3f\n' % ((end - start) // 60, (end - start) % 60,
                                                            epoch,
                                                            train_loss,
                                                            train_score))
            start = time.time()

            # Val
            if epoch == 1 or (epoch % validation_interval == 0):
                excel_validation_log_path = '%s%s_.csv' % (f"./{self.log_path}/",
                                                           f"validation_{self.trainer_params['config_name']}")
                if (self.start_epoch - 1) == 1:
                    print(f'generate {excel_validation_log_path}')
                    with open(excel_validation_log_path, 'w') as f:
                        f.write('epoch,cost,variant\n')

                val_problems = ["MDVRP", "MDOVRP", "MDVRPB", "MDVRPL", "MDVRPTW", "MDOVRPTW",
                            "MDOVRPB", "MDOVRPL", "MDVRPBL", "MDVRPBTW", "MDVRPLTW", "MDOVRPBL",
                             "MDOVRPBTW", "MDOVRPLTW", "MDVRPBLTW", "MDOVRPBLTW", "MDVRPI", "MDVRPIB",
                             "MDVRPIL", "MDVRPITW", "MDVRPIBL", "MDVRPIBTW", "MDVRPILTW", "MDVRPIBLTW"]
                val_episodes, problem_size = 1000, self.env_params['problem_size']
                model_validation_scores = {}

                for prob in val_problems:
                    dir = os.path.join("../data", prob)
                    path = "{}{}_uniform.pkl".format(prob.lower(), problem_size)
                    score = self._val_and_stat(dir, path, batch_size=250,
                                               val_episodes=val_episodes, task_name=prob)
                    model_validation_scores[prob] = score

                    # save results to a CSV file
                    with open(excel_validation_log_path, 'a') as f:
                        f.write('%d,%1.3f,%s\n' % (epoch,
                                                   score,
                                                   prob))

                    self.result_log.append('val_score', epoch, score)

            if self.env_params['curriculum'] and epoch == int(self.trainer_params['epochs'] * self.env_params['curriculum_schedule'][0]):
                # self.task_set.extend(["MDOVRPTW",
                #             "MDOVRPB", "MDOVRPL", "MDVRPBL", "MDVRPBTW", "MDVRPLTW", "MDVRPIB",
                #              "MDVRPIL", "MDVRPITW"])
                self.task_set.extend(["MDOVRPB", "MDOVRPL", "MDVRPBL", "MDVRPLTW", "MDVRPIB",
                                      "MDVRPIL"])
            elif self.env_params['curriculum'] and epoch == int(
                    self.trainer_params['epochs'] * self.env_params['curriculum_schedule'][1]):
                self.task_set.extend(["MDOVRPBL",
                                      "MDOVRPBTW", "MDOVRPLTW", "MDVRPBLTW", "MDVRPIBL", "MDVRPIBTW", "MDVRPILTW"])
            elif self.env_params['curriculum'] and epoch == int(
                    self.trainer_params['epochs'] * self.env_params['curriculum_schedule'][2]):
                self.task_set.extend(["MDOVRPBLTW", "MDVRPIBLTW"])
            num_tasks = len(self.task_set)
            uniform_prob = 1.0 / num_tasks
            self.task_probs = {task: uniform_prob for task in self.task_set}

            # Logs & Checkpoint
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(epoch, self.trainer_params['epochs'])
            self.logger.info("Epoch {:3d}/{:3d}({:.2f}%): Time Est.: Elapsed[{}], Remain[{}]".format(
                epoch, self.trainer_params['epochs'], epoch / self.trainer_params['epochs'] * 100, elapsed_time_str, remain_time_str))

            all_done = (epoch == self.trainer_params['epochs'])

            # Save Model
            if all_done or (epoch % model_save_interval) == 0:
                self.logger.info("Saving trained_model")
                checkpoint_dict = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'result_log': self.result_log.get_raw_data()
                }
                torch.save(checkpoint_dict, '{}/checkpoint-{}.pt'.format(self.result_folder, epoch))

            if all_done:
                self.logger.info(" *** Training Done *** ")
                # self.logger.info("Now, printing log array...")
                # util_print_log_array(self.logger, self.result_log)

    def _train_one_epoch(self, epoch):
        score_AM = AverageMeter()
        loss_AM = AverageMeter()

        train_num_episode = self.trainer_params['train_episodes']
        episode = 0
        loop_cnt = 0

        while episode < train_num_episode:
            remaining = train_num_episode - episode
            batch_size = min(self.trainer_params['train_batch_size'], remaining)

            # Sample one task with probability weights
            weights = [self.task_probs[task] for task in self.task_set]
            selected = random.choices(self.task_set, weights=weights, k=1)[0]

            avg_score, avg_loss = self._train_one_batch(batch_size, selected)
            episode += batch_size

            score_AM.update(avg_score, batch_size)
            loss_AM.update(avg_loss, batch_size)

            self.logger.info('Epoch {:3d}: Train {:3d}/{:3d}({:1.1f}%)  Score: {:.4f},  Loss: {:.4f}, Variant: {}'
                             .format(epoch, episode, train_num_episode, 100. * episode / train_num_episode,
                                     score_AM.avg, loss_AM.avg, selected))


        # Log Once, for each epoch
        self.logger.info('Epoch {:3d}: Train ({:3.0f}%)  Score: {:.4f},  Loss: {:.4f}'
                         .format(epoch, 100. * episode / train_num_episode,
                                 score_AM.avg, loss_AM.avg))

        return score_AM.avg, loss_AM.avg

    def _train_one_batch(self, batch_size, selected):

        env_params = self.env_params
        env_params['problem_type'] = selected

        env = Env(**env_params)
        self.model.train()
        env.load_problems(batch_size)
        reset_state, _, _ = env.reset()
        self.model.pre_forward(reset_state)
        prob_list = torch.zeros(size=(batch_size, env.pomo_size, 0))
        # shape: (batch, pomo, 0~problem)

        # POMO Rollout
        state, reward, done = env.pre_step()
        while not done:
            if self.trainer_params['use_autocast']:
                with autocast('cuda' if self.trainer_params['use_cuda'] else 'cpu', dtype=torch.float16):  # <── FP16 forward
                    selected, prob = self.model(state)
            else:
                selected, prob = self.model(state)
            # shape: (batch, pomo)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # Loss
        advantage = reward - reward.float().mean(dim=1, keepdims=True)
        # shape: (batch, pomo)
        log_prob = prob_list.log().sum(dim=2)
        # size = (batch, pomo)
        loss = -advantage * log_prob  # Minus Sign: To Increase REWARD
        # shape: (batch, pomo)
        loss_mean = loss.mean()

        #if hasattr(self.model, "aux_loss"):
        #    loss_mean = loss_mean + self.model.aux_loss  # add aux(moe)_loss for load balancing (default coefficient: 1e-2)

        # update model
        if self.trainer_params['use_autocast']:
            self.optimizer.zero_grad(set_to_none=True)

            # backward with gradient scaling
            self.scaler.scale(loss_mean).backward()
            self.scaler.unscale_(self.optimizer)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.zero_grad()
            loss_mean.backward()
            self.optimizer.step()

        # Score
        max_pomo_reward, _ = reward.max(dim=1)  # get best results from pomo
        score_mean = -max_pomo_reward.float().mean().item()  # negative sign to make positive value
        # print(score_mean)

        del env, reset_state, state, reward, done, prob_list, loss, log_prob, advantage

        return score_mean, loss_mean.item()

    def _val_one_batch(self, data, env, aug_factor=1):
        self.model.eval()
        batch_size = len(data[0])
        with torch.no_grad():
            env.load_problems(batch_size, aug_factor=aug_factor, problems=data)
            reset_state, _, _ = env.reset()
            self.model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = self.model(state)
                # shape: (batch, pomo)
                state, reward, done = env.step(selected)

        # Return
        aug_reward = reward.reshape(aug_factor, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value

        return no_aug_score, aug_score

    def _val_and_stat(self, dir, val_path, batch_size=32, val_episodes=1000, task_name=None):
        torch.cuda.empty_cache()
        no_aug_score_list, aug_score_list, no_aug_gap_list, aug_gap_list = [], [], [], []
        episode, no_aug_score, aug_score = 0, torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)

        while episode < val_episodes:
            remaining = val_episodes - episode
            bs = min(batch_size, remaining)
            data = self.env.load_dataset(os.path.join(dir, val_path), offset=episode, num_samples=bs,
                                         device=self.device)
            no_aug, aug = self._val_one_batch(data, self.env, aug_factor=8)
            no_aug_score = torch.cat((no_aug_score, no_aug), dim=0)
            aug_score = torch.cat((aug_score, aug), dim=0)
            episode += bs

        no_aug_score_list.append(round(no_aug_score.mean().item(), 4))
        aug_score_list.append(round(aug_score.mean().item(), 4))

        baseline_score = self.validation_objectives.get(task_name, None)
        gap = (aug_score_list[0] - baseline_score) / baseline_score * 100

        print(f">> Val Score on {task_name}: NO_AUG_Score: {no_aug_score_list[0]} --> AUG_Score: {aug_score_list[0]} | Baseline_Gap: {gap:.2f}%")
        return aug_score_list[0]
