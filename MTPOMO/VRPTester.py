import random
import torch
import copy

import os
from logging import getLogger

from MDVRPEnv import MDVRPEnv
from VRPEnv import VRPEnv
from VRPModel import VRPModel as Model
from VRPFineTuner import VRPFineTuner as Tuner

from utils.utils import *


class VRPTester:
    def __init__(self,
                 env_params,
                 model_params,
                 tester_params,
                 is_benchmark=False):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params
        self.num_depots = self.env_params.get('depot_size', False)
        self.is_benchmark = is_benchmark

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        if self.is_benchmark:
            if 'CVRP-LIB' in self.tester_params['dataset_path']:
                self.env = VRPEnv(**self.env_params)
            else:
                self.env = MDVRPEnv(**self.env_params)
        else:
            if self.num_depots > 1:
                self.env = MDVRPEnv(**self.env_params)
            else:
                self.env = VRPEnv(**self.env_params)
        if self.tester_params.get('sample_size', 1) > 1:
            self.model_params['eval_type'] = 'softmax'
        self.model = Model(**self.model_params)

        if not self.is_benchmark:
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
            else:
                self.task_set = [self.env_params['problem_type']]

            if self.tester_params['augmentation_type'] == '1':
                self.tester_params['aug_factor'] = 1
            elif self.tester_params['augmentation_type'] == '8':
                self.tester_params['aug_factor'] = 8
            elif self.tester_params['augmentation_type'] == 'd':
                self.tester_params['aug_factor'] = self.num_depots + 7
            elif self.tester_params['augmentation_type'] == '8d':
                self.tester_params['aug_factor'] = 8 * self.num_depots
            else:
                raise NotImplementedError

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])

        # utility
        self.time_estimator = TimeEstimator()

    def run(self):
        self.time_estimator.reset()

        test_num_episode = 1000 if not self.tester_params['random_problems'] else self.tester_params[
            'num_random_problems']

        for task in self.task_set:
            start = time.time()
            no_aug_score_list, aug_score_list = [], []
            episode, no_aug_score, aug_score = 0, torch.zeros(0).to(self.device), torch.zeros(0).to(self.device)

            if not self.tester_params['random_problems']:
                with open(f"../data/pyvrp_results_vrp{self.env_params['problem_size']}.json", "r") as f:
                    self.validation_objectives = load_baselines(self.env_params["problem_size"], f"../data/{task}",
                                                                task.lower())

            if self.tester_params['enable_finetuning']:
                tuner = Tuner(self.env_params, self.model_params, self.tester_params['tuner_params'], self.model, task)
                tuner.run()
                model = Model(**self.model_params)
                model.load_state_dict(copy.deepcopy(tuner.model.state_dict()))
            else:
                model = copy.deepcopy(self.model)

            if not self.tester_params['random_problems']:
                task_path = "{}{}_uniform.pkl".format(task.lower(), self.env_params['problem_size'])
                dir = os.path.join("../data", task)

                while episode < test_num_episode:
                    remaining = test_num_episode - episode
                    batch_size = min(self.tester_params['test_batch_size'], remaining)
                    if self.num_depots > 1:
                        data = self.env.load_dataset(os.path.join(dir, task_path), offset=episode,
                                                     num_samples=batch_size,
                                                     device=self.device)
                    else:
                        data = self.env.load_dataset(os.path.join(dir, task_path), offset=episode,
                                                     num_samples=batch_size,
                                                     device=self.device, task=task)
                    no_aug, aug, _ = self._test_one_batch(model, data, self.env,
                                                       aug_factor=self.tester_params['aug_factor'])
                    no_aug_score = torch.cat((no_aug_score, no_aug), dim=0)
                    aug_score = torch.cat((aug_score, aug), dim=0)
                    episode += batch_size
            else:
                while episode < test_num_episode:
                    remaining = test_num_episode - episode
                    batch_size = min(self.tester_params['test_batch_size'], remaining)
                    no_aug, aug = self._test_one_batch_random(model, batch_size, task,
                                                              aug_factor=self.tester_params['aug_factor'])
                    no_aug_score = torch.cat((no_aug_score, no_aug), dim=0)
                    aug_score = torch.cat((aug_score, aug), dim=0)
                    episode += batch_size

            no_aug_mean = round(no_aug_score.mean().item(), 3)
            aug_mean = round(aug_score.mean().item(), 3)

            no_aug_score_list.append(no_aug_mean)
            aug_score_list.append(aug_mean)

            if not self.tester_params['random_problems']:
                # Compute instance-level gaps
                best_objs = get_best_baselines(self.validation_objectives)
                best_objs_tensor = torch.tensor(best_objs, device=self.device)

                no_aug_gap_inst = ((no_aug_score - best_objs_tensor) / best_objs_tensor) * 100
                aug_gap_inst = ((aug_score - best_objs_tensor) / best_objs_tensor) * 100

                # Mean gap
                no_aug_gap_mean = no_aug_gap_inst.mean().item()
                aug_gap_mean = aug_gap_inst.mean().item()

                print_baseline_gaps(self.validation_objectives, best_objs, task)
                print(f">> Test Score on {task}: "
                      f"NO_AUG_Mean: {no_aug_score.mean().item():.3f} | NO_AUG_Gap: {no_aug_gap_mean:.3f}% "
                      f"--> AUG_Mean: {aug_score.mean().item():.3f} | AUG_Gap: {aug_gap_mean:.3f}% "
                      f"--> Elapsed Time: {(time.time() - start):.3f} seconds\n")
            else:
                print(
                    ">> Test Score on {}: NO_AUG_Score: {}, --> AUG_Score: {} --> Elapsed Time: {:.3f} seconds\n".format(
                        task, no_aug_score_list, aug_score_list, (time.time() - start)))

    def _test_one_batch(self, model, data, env, aug_factor=1, is_benchmark=False):
        model.eval()
        batch_size = len(data[0])
        sample_size = self.tester_params.get('sample_size', 1)
        if self.model_params['eval_type'] == "softmax":
            test_data = list(data)
            for i, data in enumerate(test_data):
                if data.dim() == 1:
                    test_data[i] = data.repeat(sample_size)
                elif data.dim() == 2:
                    test_data[i] = data.repeat(sample_size, 1)
                elif data.dim() == 3:
                    test_data[i] = data.repeat(sample_size, 1, 1)
        else:
            test_data = data


        with torch.no_grad():
            env.load_problems(batch_size * sample_size, aug_factor=aug_factor, problems=test_data, is_benchmark=is_benchmark)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            state, reward, done = env.pre_step()

            while not done:
                selected, _ = model(state)
                # shape: (batch, pomo)
                state, reward, done = env.step(selected)

        # Return
        aug_reward = reward.reshape(aug_factor * sample_size, batch_size, env.pomo_size)
        # shape: (augmentation, batch, pomo)
        max_pomo_reward, _ = aug_reward.max(dim=2)  # get best results from pomo
        no_aug_score = -max_pomo_reward[0, :].float()  # negative sign to make positive value
        max_aug_pomo_reward, _ = max_pomo_reward.max(dim=0)  # get best results from augmentation
        aug_score = -max_aug_pomo_reward.float()  # negative sign to make positive value

        return no_aug_score, aug_score, aug_reward

    def _test_one_batch_random(self, model, batch_size, selected, aug_factor=1):
        env_params = self.env_params
        env_params['problem_type'] = selected

        if self.num_depots > 1:
            env = MDVRPEnv(**env_params)
        else:
            env = VRPEnv(**env_params)

        model.eval()
        with torch.no_grad():
            env.load_problems(batch_size, aug_factor=aug_factor)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)
            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
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

    def _solve_cvrplib_(self):
        """
            Solve CVRPLIB dataset.
        """
        self.time_estimator.reset()

        start_time = time.time()
        no_aug_gap_list, aug_gap_list, ls_gap_list = [], [], []

        path_list = [os.path.join(self.tester_params['dataset_path'], f) for f in
                          sorted(os.listdir(self.tester_params['dataset_path']))] \
            if os.path.isdir(self.tester_params['dataset_path']) else [self.tester_params['dataset_path']]
        assert path_list[-1].endswith(".vrp") or path_list[-1].endswith(".txt"), "Unsupported file types."

        for path in path_list:
            if not path.endswith(".sol"):
                file = open(path, "r")
                lines = [ll.strip() for ll in file]
                i = 0
                if 'Solomon' in path:
                    while i < len(lines):
                        line = lines[i]
                        if line.startswith("NUMBER"):
                            line = lines[i + 1]
                            capacity = int(line.split(' ')[-1])
                        elif line.startswith("CUST NO."):
                            data = np.loadtxt(lines[i + 1:], dtype=int)
                            break
                        i += 1
                    # --- Automatic shift + scale ---
                    original_locations = data[:, 1:3]  # remove node index
                    original_locations = np.expand_dims(original_locations, axis=0)  # [1, n+1, 2]
                    scale = max(original_locations.max(),
                                 data[0, 5] / 3.)  # we set the time window of the depot node as [0, 3]
                    assert original_locations.max() <= scale, ">> Scaler is too small for {}".format(path)
                    locations = original_locations / scale  # [1, n+1, 2]: Scale location coordinates to [0, 1]
                    depot_xy, node_xy = torch.Tensor(locations[:, :1, :]), torch.Tensor(locations[:, 1:, :])
                    node_demand = torch.Tensor(data[1:, 3].reshape((1, -1))) / capacity  # [1, n]
                    service_time = torch.Tensor(data[1:, -1].reshape((1, -1))) / scale  # [1, n]
                    tw_start = torch.Tensor(data[1:, 4].reshape((1, -1))) / scale  # [1, n]
                    tw_end = torch.Tensor(data[1:, 5].reshape((1, -1))) / scale  # [1, n]
                    data = (depot_xy, node_xy, node_demand, service_time, tw_start, tw_end)
                else:
                    while i < len(lines):
                        line = lines[i]
                        if line.startswith("DIMENSION"):
                            dimension = int(line.split(':')[1])
                        elif line.startswith("CAPACITY"):
                            capacity = int(line.split(':')[1])
                        elif line.startswith('NODE_COORD_SECTION'):
                            locations = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                            i = i + dimension
                        elif line.startswith('DEMAND_SECTION'):
                            demand = np.loadtxt(lines[i + 1:i + 1 + dimension], dtype=int)
                            i = i + dimension
                        i += 1
                    # --- Automatic shift + scale ---
                    original_locations = locations[:, 1:]  # remove node index

                    # 1. Shift negatives slightly above 0
                    min_xy = original_locations.min(axis=0)
                    shift = np.maximum(-min_xy + 0.01, 0)
                    locations_shifted = original_locations + shift

                    # 2. Scale to roughly [0,1]
                    scale = locations_shifted.max()
                    locations_scaled = locations_shifted / scale

                    # 3. Convert to tensors
                    depot_xy = torch.tensor(locations_scaled[:1, :], dtype=torch.float32).unsqueeze(0).to(self.device)
                    node_xy = torch.tensor(locations_scaled[1:, :], dtype=torch.float32).unsqueeze(0).to(self.device)
                    node_demand = torch.Tensor(demand[1:, 1:].reshape((1, -1))) / capacity  # [1, n]
                    data = (depot_xy, node_xy, node_demand)

                if self.tester_params['augmentation_type'] == '1':
                    aug_factor = 1
                elif self.tester_params['augmentation_type'] == '8':
                    aug_factor = 8
                elif self.tester_params['augmentation_type'] == 'd':
                    aug_factor = 1 + 7
                elif self.tester_params['augmentation_type'] == '8d':
                    aug_factor = 8 * 1
                else:
                    raise NotImplementedError

                no_aug_score, aug_score, aug_reward = self._test_one_batch(copy.deepcopy(self.model), data, self.env, aug_factor=aug_factor, is_benchmark=True)
                no_aug_score = (no_aug_score * scale).item()
                aug_score = (aug_score * scale).item()

                sol_path = path.replace(".vrp", ".sol").replace(".txt", ".sol")
                bks_cost = None
                if os.path.exists(sol_path):
                    with open(sol_path, "r") as f:
                        for line in f:
                            if line.startswith("Cost"):
                                bks_cost = float(line.split()[1])
                                break

                if bks_cost is not None:
                    no_aug_gap = (no_aug_score - bks_cost) / bks_cost * 100
                    aug_gap = (aug_score - bks_cost) / bks_cost * 100
                    no_aug_gap_list.append(no_aug_gap)
                    aug_gap_list.append(aug_gap)
                    no_aug_gap = round(no_aug_gap, 3)
                    aug_gap = round(aug_gap, 3)
                else:
                    no_aug_gap = aug_gap = None

                print(f">> Test Score on {path} -> \n"
                        f"NO_AUG: {round(no_aug_score, 3)} | NO_AUG_GAP: {no_aug_gap if no_aug_gap is not None else 'N/A'}%\n"
                        f"AUG: {round(aug_score, 3)} | AUG_GAP: {aug_gap if aug_gap is not None else 'N/A'}%\n"
                        f"BKS: {bks_cost if bks_cost is not None else 'N/A'}\n")

        # --- elapsed time ---
        elapsed_time = time.time() - start_time

        # --- mean gaps ---
        mean_no_aug_gap = round(np.mean(no_aug_gap_list), 3) if no_aug_gap_list else None
        mean_aug_gap = round(np.mean(aug_gap_list), 3) if aug_gap_list else None

        print(f"\n>> Elapsed time: {elapsed_time:.2f}s")
        print(f">> Mean NO_AUG gap: {mean_no_aug_gap if mean_no_aug_gap is not None else 'N/A'}%")
        print(f">> Mean AUG gap: {mean_aug_gap if mean_aug_gap is not None else 'N/A'}%")

    def _solve_mdvrplib_(self):
        """
            Solve Cordeau's dataset.
        """
        self.time_estimator.reset()

        start_time = time.time()
        no_aug_gap_list, aug_gap_list, ls_gap_list = [], [], []

        path_list = [os.path.join(self.tester_params['dataset_path'], f) for f in
                          sorted(os.listdir(self.tester_params['dataset_path']))] \
            if os.path.isdir(self.tester_params['dataset_path']) else [self.tester_params['dataset_path']]
        
        # Detect variant features from dataset path (e.g., "Cordeau-MDOVRPTW")
        dataset_path = self.tester_params['dataset_path']
        # Extract variant name from path (e.g., "MDOVRPTW" from "Cordeau-MDOVRPTW")
        variant_name = ""
        for part in dataset_path.replace("\\", "/").split("/"):
            if part.startswith("Cordeau-"):
                variant_name = part.replace("Cordeau-", "")
                break
        
        # Detect O (Open routes) - check if variant contains "O" after "MD"
        has_open = "MDOVRP" in variant_name or "MDO" in variant_name
        # Detect I (Inter-depot) - check if variant contains "I"
        has_interdepot = "I" in variant_name and "MDVRPI" in variant_name

        for path in path_list:
            if not path.endswith(".res"):
                file = open(path, "r")
                lines = [ll.strip() for ll in file]
                i = 0
                route_length_limits = []
                capacities = []
                locations = []
                demands = []
                service_durations = []
                early_tws = []
                late_tws = []
                while i < len(lines):
                    line = tuple(map(lambda z: float(z), lines[i].strip().split()))
                    if i == 0:
                        num_depots = int(line[-1])
                        num_customers = int(line[2])
                    elif i < 1 + num_depots:
                        route_length_limits.append(line[0])
                        capacities.append(line[1])
                    elif i < 1 + num_depots + num_customers:
                        locations.append([line[1], line[2]])
                        demands.append(line[4])
                        if 'TW' in path:
                            service_durations.append(line[3])
                            early_tws.append(line[-2])
                            late_tws.append(line[-1])
                        else:
                            service_durations.append(0)
                            early_tws.append(0)
                            late_tws.append(0)
                    else:
                        locations.append([line[1], line[2]])
                    i += 1
                locations = np.array(locations)
                demand = np.array(demands)
                service_durations = np.array(service_durations)
                early_tws = np.array(early_tws)
                late_tws = np.array(late_tws)
                capacity = capacities[0]

                # --- Automatic shift + scale ---
                original_locations = locations.copy()

                # 1. Shift so min is 0
                shift = original_locations.min(axis=0)  # shape (2,) if 2D
                locations_shifted = original_locations - shift

                # 2. Scale to [0,1]
                scale = locations_shifted.max()
                locations_scaled = locations_shifted / scale

                # 3. Convert to tensors
                depot_xy = torch.tensor(locations_scaled[-num_depots:, :], dtype=torch.float32).unsqueeze(0).to(self.device)
                node_xy = torch.tensor(locations_scaled[:-num_depots, :], dtype=torch.float32).unsqueeze(0).to(self.device)
                node_demand_scaled = torch.Tensor(demand.reshape((1, -1))) / capacity  # [1, n]
                node_sds_scaled = torch.Tensor(service_durations.reshape(1, -1)) / scale
                early_tws_scaled = torch.Tensor(early_tws.reshape(1, -1)) / scale
                late_tws_scaled = torch.Tensor(late_tws.reshape(1, -1)) / scale
                route_lengths_scaled = torch.zeros(size=(1, num_customers))
                route_lengths_scaled.fill_(route_length_limits[0] / scale)
                
                # Create is_open and is_interdepot tensors based on variant detection
                is_open = torch.ones(size=(1, num_customers)) if has_open else torch.zeros(size=(1, num_customers))
                is_interdepot = torch.ones(size=(1, num_customers)) if has_interdepot else torch.zeros(size=(1, num_customers))
                
                # Data tuple now includes is_open and is_interdepot
                data = (depot_xy, node_xy, node_demand_scaled, route_lengths_scaled, 
                        node_sds_scaled, early_tws_scaled, late_tws_scaled, 
                        is_open, is_interdepot)

                if self.tester_params['augmentation_type'] == '1':
                    aug_factor = 1
                elif self.tester_params['augmentation_type'] == '8':
                    aug_factor = 8
                elif self.tester_params['augmentation_type'] == 'd':
                    aug_factor = num_depots + 7
                elif self.tester_params['augmentation_type'] == '8d':
                    aug_factor = 8 * num_depots
                else:
                    raise NotImplementedError

                no_aug_score, aug_score, aug_reward = self._test_one_batch(copy.deepcopy(self.model), data, self.env, aug_factor=aug_factor, is_benchmark=True)
                no_aug_score = (no_aug_score * scale).item()
                aug_score = (aug_score * scale).item()

                root, ext = os.path.splitext(path)  # splits "file.ext" into ("file", ".ext")
                sol_path = root + ".res"
                bks_cost = None
                if os.path.exists(sol_path):
                    with open(sol_path, "r") as f:
                        first_line = f.readline().strip()  # read the first line
                        bks_cost = float(first_line)  # convert it to float

                # Compute gaps against BKS if available
                if bks_cost is not None:
                    no_aug_gap = (no_aug_score - bks_cost) / bks_cost * 100
                    aug_gap = (aug_score - bks_cost) / bks_cost * 100
                    no_aug_gap_list.append(no_aug_gap)
                    aug_gap_list.append(aug_gap)
                    no_aug_gap = round(no_aug_gap, 3)
                    aug_gap = round(aug_gap, 3)
                else:
                    no_aug_gap = aug_gap = None

                print(f">> Test Score on {path} -> \n"
                        f"NO_AUG: {round(no_aug_score, 3)} | NO_AUG_GAP: {no_aug_gap if no_aug_gap is not None else 'N/A'}%\n"
                        f"AUG: {round(aug_score, 3)} | AUG_GAP: {aug_gap if aug_gap is not None else 'N/A'}%\n"
                        f"BKS: {bks_cost if bks_cost is not None else 'N/A'}\n")

        # --- elapsed time ---
        elapsed_time = time.time() - start_time

        # --- mean gaps ---
        mean_no_aug_gap = round(np.mean(no_aug_gap_list), 3) if no_aug_gap_list else None
        mean_aug_gap = round(np.mean(aug_gap_list), 3) if aug_gap_list else None

        print(f"\n>> Elapsed time: {elapsed_time:.2f}s")
        print(f">> Mean NO_AUG gap: {mean_no_aug_gap if mean_no_aug_gap is not None else 'N/A'}%")
        print(f">> Mean AUG gap: {mean_aug_gap if mean_aug_gap is not None else 'N/A'}%")
