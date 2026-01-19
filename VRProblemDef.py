import os
import copy
import torch
import pickle
import numpy as np


def get_random_problems_mixed(batch_size, problem_size, depot_size, problem_type, dump=False):
    
    depot_xy = torch.rand(size=(batch_size, depot_size, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)


    # if size > 50, demand_scaler = 30 + size/5
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 200:
        demand_scaler = 70
    elif problem_size == 500:
        demand_scaler = 130
    elif problem_size == 1000:
        demand_scaler = 230
    else:
        raise NotImplementedError
        
    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    node_serviceTime = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)
    # zeros 

    node_lengthTW = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)

    node_earlyTW = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)
    # default velocity = 1.0

    node_lateTW = node_earlyTW + node_lengthTW
    # shape: (batch, problem)

    route_length_limit = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)

    route_open = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)
    
    route_interdepot = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)

    seed = np.random.rand()

    if 'L' in problem_type:
        depots_exp = depot_xy[:, :, None, :]  # [batch, depot_size, 1, 2]
        customers_exp = node_xy[:, None, :, :]  # [batch, 1, problem_size, 2]

        # Euclidean distances [batch, depot_size, problem_size]
        dist_depot_to_cust = (depots_exp - customers_exp).norm(p=2, dim=-1)

        # For each depot, get max distance to customers: [batch, depot_size]
        max_dist_per_depot, _ = dist_depot_to_cust.max(dim=2)

        # For each batch, get max of max distances across depots: [batch]
        max_dist = max_dist_per_depot.max(dim=1).values

        # Calculate lower bound: 2 * max_dist: [batch]
        lower_bound = 2 * max_dist

        # Sample uniform random numbers between lower_bound and 3 for each batch and problem_size
        # We want route_length_limit: [batch, problem_size]
        route_length_limit = torch.rand(batch_size) * (3 - lower_bound) + lower_bound  # [batch]

        # If you want to broadcast it to [batch, pomo_size] or [batch, problem_size]:
        # For example, if you want [batch, pomo_size]:
        route_length_limit = route_length_limit[:, None].expand(batch_size, problem_size)

    if 'TW' in problem_type:
        a, b, c = 0.15, 0.18, 0.2
        node_serviceTime = a + (b - a) * torch.rand(batch_size, problem_size)

        locs = torch.cat((depot_xy, node_xy), dim=1)
        tensor_a = locs[:, 0: depot_size]  # Shape: [B, num_depots, 2]
        tensor_b = locs[:, depot_size:]  # Shape: [B, N, 2]

        tensor_a_expanded = tensor_a[:, :, None, :]  # Shape: [B, num_depots, 1, 2]
        tensor_b_expanded = tensor_b[:, None, :, :]  # Shape: [B, 1, N, 2]

        d_0i = (tensor_a_expanded - tensor_b_expanded).norm(p=2, dim=-1)
        # Shape: [B, num_depots, N]

        d_0i, _ = torch.max(d_0i, dim=1)  # Shape: [B, N]

        tw_length = b + (c - b) * torch.rand(batch_size, problem_size)
        h_max = (4.6 - node_serviceTime - tw_length) / d_0i * 1 - 1
        node_earlyTW = (1 + (h_max - 1) * torch.rand(batch_size, problem_size)) * d_0i / 1
        node_lateTW = node_earlyTW + tw_length

    if 'O' in problem_type:

        route_open = torch.ones(size=(batch_size, problem_size))
        # shape: (batch, problem)
    
    if 'I' in problem_type:

        route_interdepot = torch.ones(size=(batch_size, problem_size + depot_size))
        # shape: (batch, problem)

    if 'B' in problem_type:

        backhauls_index = torch.randperm(problem_size)[
                          :int(problem_size * 0.2)]  # randomly select 20% customers as backhaul ones
        node_demand[:, backhauls_index] = -1 * node_demand[:, backhauls_index]

    if dump:
        return {'depot_xy': depot_xy,
                'node_xy': node_xy,
                'node_demand': node_demand,
                'node_earlyTW': node_earlyTW,
                'node_lateTW': node_lateTW,
                'node_serviceTime': node_serviceTime,
                'route_open': route_open,
                'route_interdepot': route_interdepot,
                'route_length_limit': route_length_limit}
    else:
        return depot_xy, node_xy, node_demand, node_earlyTW, node_lateTW, node_serviceTime, route_open, route_interdepot, route_length_limit


def get_random_problems_mixed_single_depot(batch_size, problem_size, depot_size, problem_type, dump=False):
    depot_xy = torch.rand(size=(batch_size, depot_size, 2))
    # shape: (batch, 1, 2)

    node_xy = torch.rand(size=(batch_size, problem_size, 2))
    # shape: (batch, problem, 2)

    # if size > 50, demand_scaler = 30 + size/5
    if problem_size == 20:
        demand_scaler = 30
    elif problem_size == 50:
        demand_scaler = 40
    elif problem_size == 100:
        demand_scaler = 50
    elif problem_size == 200:
        demand_scaler = 70
    elif problem_size == 500:
        demand_scaler = 130
    elif problem_size == 1000:
        demand_scaler = 230
    else:
        raise NotImplementedError

    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(demand_scaler)
    # shape: (batch, problem)

    node_serviceTime = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)
    # zeros

    node_lengthTW = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)

    node_earlyTW = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)
    # default velocity = 1.0

    node_lateTW = node_earlyTW + node_lengthTW
    # shape: (batch, problem)

    route_length_limit = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)

    route_open = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)

    route_interdepot = torch.zeros(size=(batch_size, problem_size))
    # shape: (batch, problem)

    seed = np.random.rand()

    if 'L' in problem_type:
        route_length_limit = torch.ones(batch_size) * 3.0
        route_length_limit = route_length_limit[:, None].expand(batch_size, problem_size)

    if 'O' in problem_type:
        route_open = torch.ones(size=(batch_size, problem_size))
        # shape: (batch, problem)

    if 'B' in problem_type:
        backhauls_index = torch.randperm(problem_size)[
            :int(problem_size * 0.2)]  # randomly select 20% customers as backhaul ones
        node_demand[:, backhauls_index] = -1 * node_demand[:, backhauls_index]

    if 'TW' in problem_type:
        #   2. See "Learning to Delegate for Large-scale Vehicle Routing" in NeurIPS 2021.
        #   Note: this setting follows a similar procedure as in Solomon, and therefore is more realistic and harder.
        node_serviceTime = torch.ones(batch_size, problem_size) * 0.2
        travel_time = (node_xy - depot_xy).norm(p=2, dim=-1) / 1
        a, b = 0 + travel_time, 3 - travel_time - node_serviceTime
        time_centers = (a - b) * torch.rand(batch_size, problem_size) + b
        time_half_width = (node_serviceTime / 2 - 3 / 3) * torch.rand(batch_size,
                                                                               problem_size) + 3 / 3
        node_earlyTW = torch.clamp(time_centers - time_half_width, min=0, max=3)
        node_lateTW = torch.clamp(time_centers + time_half_width, min=0, max=3)
        # shape: (batch, problem)

        # check tw constraint: feasible solution must exist (i.e., depot -> a random node -> depot must be valid).
        instance_invalid, round_error_epsilon = False, 0.00001
        total_time = torch.max(0 + (depot_xy - node_xy).norm(p=2, dim=-1) / 1, node_earlyTW) + node_serviceTime + (
                    node_xy - depot_xy).norm(p=2, dim=-1) / 1 > 3 + round_error_epsilon
        # (batch, problem)
        instance_invalid = total_time.any()

        if instance_invalid:
            print(">> Invalid instances, Re-generating ...")
            return get_random_problems_mixed_single_depot(batch_size, problem_size, depot_size, problem_type, dump)

    if dump:
        return {'depot_xy': depot_xy,
                'node_xy': node_xy,
                'node_demand': node_demand,
                'node_earlyTW': node_earlyTW,
                'node_lateTW': node_lateTW,
                'node_serviceTime': node_serviceTime,
                'route_open': route_open,
                'route_interdepot': route_interdepot,
                'route_length_limit': route_length_limit}
    else:
        return depot_xy, node_xy, node_demand, node_earlyTW, node_lateTW, node_serviceTime, route_open, route_interdepot, route_length_limit


def augment_xy_data(xy_data, aug_factor, depot_size, is_depot):
    
    # Original POMO augmentation (Kwon et al., 2020), considers that the initial depot assigment is always 
    # the one with index 0
    if aug_factor == 8:
        # xy_data.shape: (batch, N, 2)
        
        x = xy_data[:, :, [0]]
        y = xy_data[:, :, [1]]
        # x,y shape: (batch, N, 1)

        data1 = torch.cat((x, y), dim=2)
        data2 = torch.cat((1 - x, y), dim=2)
        data3 = torch.cat((x, 1 - y), dim=2)
        data4 = torch.cat((1 - x, 1 - y), dim=2)
        data5 = torch.cat((y, x), dim=2)
        data6 = torch.cat((1 - y, x), dim=2)
        data7 = torch.cat((y, 1 - x), dim=2)
        data8 = torch.cat((1 - y, 1 - x), dim=2)

        aug_xy_data = torch.cat((data1, data2, data3, data4, data5, data6, data7, data8), dim=0)
        # shape: (8*batch, N, 2)
        
    # Multi-depot multi-type attention (Jinqi Li et al., 2024) augmentation, considers original POMO augmentation +
    # (depot_size - 1) instances from which vehicles begin
    elif aug_factor == (7 + depot_size):
        # xy_data.shape: (batch, N, 2)
        
        x = xy_data[:, :, [0]]
        y = xy_data[:, :, [1]]
        # x,y shape: (batch, N, 1)

        data1 = torch.cat((x, y), dim=2)
        data2 = torch.cat((1 - x, y), dim=2)
        data3 = torch.cat((x, 1 - y), dim=2)
        data4 = torch.cat((1 - x, 1 - y), dim=2)
        data5 = torch.cat((y, x), dim=2)
        data6 = torch.cat((1 - y, x), dim=2)
        data7 = torch.cat((y, 1 - x), dim=2)
        data8 = torch.cat((1 - y, 1 - x), dim=2)

        aug_xy_data = torch.cat((data1, data2, data3, data4, data5, data6, data7, data8), dim=0)
        # shape: (8*batch, N, 2)
        
        for i in range(depot_size - 1):
            x_y_temp = copy.deepcopy(xy_data)
            if is_depot:
                x_y_temp[:, 0] = x_y_temp[:, i + 1]
                x_y_temp[:, i + 1] = xy_data[:, 0]
            
            aug_xy_data = torch.cat((aug_xy_data, x_y_temp), dim=0)
            # final shape: ((depot_size + 7)*batch, N, 2)
            
    # Full depot-wise POMO augmentation (8 augmentations for each depot)
    elif aug_factor == (8 * depot_size):
        aug_xy_data_list = []

        for i in range(depot_size):
            x_y_temp = copy.deepcopy(xy_data)

            if is_depot:
                x_y_temp[:, 0], x_y_temp[:, i] = xy_data[:, i].clone(), xy_data[:, 0].clone()

            x = x_y_temp[:, :, [0]]
            y = x_y_temp[:, :, [1]]

            data1 = torch.cat((x, y), dim=2)
            data2 = torch.cat((1 - x, y), dim=2)
            data3 = torch.cat((x, 1 - y), dim=2)
            data4 = torch.cat((1 - x, 1 - y), dim=2)
            data5 = torch.cat((y, x), dim=2)
            data6 = torch.cat((1 - y, x), dim=2)
            data7 = torch.cat((y, 1 - x), dim=2)
            data8 = torch.cat((1 - y, 1 - x), dim=2)

            depot_aug_set = torch.cat((data1, data2, data3, data4, data5, data6, data7, data8), dim=0)
            # shape: (8*batch, N, 2)
            aug_xy_data_list.append(depot_aug_set)

        aug_xy_data = torch.cat(aug_xy_data_list, dim=0)
        # final shape: ((depot_size * 8)*batch, N, 2)

    return aug_xy_data

if __name__ == '__main__':
    #problems = ["MDVRP", "MDOVRP", "MDVRPB", "MDVRPL", "MDVRPTW", "MDOVRPTW",
    #                        "MDOVRPB", "MDOVRPL", "MDVRPBL", "MDVRPBTW", "MDVRPLTW", "MDOVRPBL", 
    #                         "MDOVRPBTW", "MDOVRPLTW", "MDVRPBLTW", "MDOVRPBLTW"]
    
    # problems = ["MDVRPI", "MDVRPIB", "MDVRPIL", "MDVRPITW", "MDVRPIBL", "MDVRPIBTW", "MDVRPILTW", "MDVRPIBLTW"]
    problems = ['MDVRP']
    # problem = 'MDVRPTW'
    num_instances = 100
    problem_size = 1000
    depot_size = 5
    
    for problem in problems:
        data = get_random_problems_mixed(num_instances, problem_size, depot_size, problem, dump=True)

        path = os.path.join("./data", problem,
                                    "{}{}_uniform.pkl".format(problem.lower(), problem_size))
        dataset = {key: value.cpu().numpy() for key, value in data.items()}
        filedir = os.path.split(path)[0]
        if not os.path.isdir(filedir):
            os.makedirs(filedir)
        with open(path, 'wb') as f:
            pickle.dump(dataset, f)
        print("Save {} dataset to {}".format(problem, path))
