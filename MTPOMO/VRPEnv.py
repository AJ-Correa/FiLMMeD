import os
import torch
import pickle
from dataclasses import dataclass

from VRProblemDef import get_random_problems_mixed_single_depot, augment_xy_data


@dataclass
class Reset_State:
    depot_xy: torch.Tensor = None
    # shape: (batch, 1, 2)
    node_xy: torch.Tensor = None
    # shape: (batch, problem, 2)
    node_demand: torch.Tensor = None
    # shape: (batch, problem)
    node_earlyTW: torch.Tensor = None
    # shape: (batch, problem)
    node_lateTW: torch.Tensor = None
    # shape: (batch, problem)
    node_serviceTime: torch.Tensor = None
    # shape: (batch, problem)
    # route_open: torch.Tensor = None
    # # shape: (batch, problem)
    # length: torch.Tensor = None
    # # shape: (batch, problem)
    is_open: torch.Tensor = None
    # shape: (batch, pomo)
    is_interdepot: torch.Tensor = None
    # shape: (batch, pomo)
    is_tw: torch.Tensor = None
    # shape: (batch, pomo)
    is_backhaul: torch.Tensor = None
    # shape: (batch, pomo)
    is_limit: torch.Tensor = None
    # shape: (batch, pomo)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    start_node: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    time: torch.Tensor = None
    # shape: (batch, pomo)
    is_open: torch.Tensor = None
    # shape: (batch, pomo)
    is_interdepot: torch.Tensor = None
    # shape: (batch, pomo)
    length: torch.Tensor = None
    # shape: (batch, pomo)

    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class VRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        ####################################
        self.env_params = env_params
        self.problem_size = env_params.get('problem_size', False)
        self.pomo_size = env_params.get('pomo_size', False)
        self.problem_type = env_params.get('problem_type', False)
        self.depot_size = env_params.get('depot_size', False)

        self.saved_depot_xy = None
        self.saved_node_xy = None
        self.saved_node_demand = None
        self.saved_index = None

        # Const @Load_Problem
        ####################################
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        self.start_node = None
        # IDX.shape: (batch, pomo)
        self.depot_node_xy = None
        # shape: (batch, problem+1, 2)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.depot_node_earlyTW = None
        # shape: (batch, problem+1)
        self.depot_node_lateTW = None
        # shape: (batch, problem+1)
        self.depot_node_servicetime = None
        # shape: (batch, problem+1)
        self.length = None
        # shape: (batch, pomo)

        ##################################
        self.attribute_c = False
        self.attribute_tw = False
        self.attribute_o = False
        self.attribute_id = False
        self.attribute_b = False
        self.attribute_l = False

        # Dynamic-1
        ####################################
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.time = None
        # shape: (batch, pomo)
        self.is_open = None
        # shape: (batch, pomo)
        self.is_interdepot = None
        # shape: (batch, pomo)
        self.is_tw = None
        # shape: (batch, pomo)
        self.is_backhaul = None
        # shape: (batch, pomo)
        self.is_limit = None
        # shape: (batch, pomo)
        self.length = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        ####################################
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size, aug_factor=1, problems=None, is_benchmark=False):
        self.batch_size = batch_size

        if problems is None:
            depot_xy, node_xy, node_demand, node_earlyTW, node_lateTW, node_servicetime, is_open, is_interdepot, route_length_limit = get_random_problems_mixed_single_depot(
                batch_size, self.problem_size, self.env_params["depot_size"], self.problem_type)
        elif is_benchmark:
            depot_xy = problems[0]
            node_xy = problems[1]
            node_demand = problems[2]

            self.problem_size = node_xy.size(1)
            self.pomo_size = self.problem_size
            self.depot_size = 1

            if len(problems) == 3:
                node_earlyTW = torch.zeros(size=(batch_size, self.problem_size))
                node_lateTW = torch.zeros(size=(batch_size, self.problem_size))
                node_servicetime = torch.zeros(size=(batch_size, self.problem_size))
            else:
                node_servicetime = problems[3]
                node_earlyTW = problems[4]
                node_lateTW = problems[5]
            is_open = torch.zeros(size=(batch_size, self.problem_size))
            is_interdepot = torch.zeros(size=(batch_size, self.problem_size))
            route_length_limit = torch.zeros(size=(batch_size, self.problem_size))
        else:
            depot_xy = problems[0]
            node_xy = problems[1]
            node_demand = problems[2]
            node_earlyTW = problems[3]
            node_lateTW = problems[4]
            node_servicetime = problems[5]
            is_open = problems[6]
            is_interdepot = problems[7]
            route_length_limit = problems[8]

        if aug_factor > 1:
            self.batch_size = self.batch_size * aug_factor
            depot_xy = augment_xy_data(depot_xy, aug_factor, self.depot_size, is_depot=True)
            node_xy = augment_xy_data(node_xy, aug_factor, self.depot_size, is_depot=False)
            node_demand = node_demand.repeat(aug_factor, 1)
            node_earlyTW = node_earlyTW.repeat(aug_factor, 1)
            node_lateTW = node_lateTW.repeat(aug_factor, 1)
            node_servicetime = node_servicetime.repeat(aug_factor, 1)
            is_open = is_open.repeat(aug_factor, 1)
            is_interdepot = is_interdepot.repeat(aug_factor, 1)
            route_length_limit = route_length_limit.repeat(aug_factor, 1)

        if self.env_params["depot_trick"]:
            # Depot trick: create virtual depots for single-depot CVRP to mimic MDVRP structure
            num_virtual_depots = self.env_params["num_virtual_depots"]

            if num_virtual_depots > 1:
                # Repeat the single depot's coordinates along the depot dimension
                depot_xy = depot_xy.repeat(1, num_virtual_depots, 1)  # [batch, num_virtual_depots, 2]

                # Update internal state to reflect virtual depots
                self.depot_size = num_virtual_depots
                self.pomo_size = self.problem_size + num_virtual_depots - 1

        self.attribute_id = False

        self.is_open = is_open
        # shape: (batch,pomo)
        self.is_interdepot = is_interdepot
        # shape: (batch,pomo)

        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1)
        # shape: (batch, problem+1, 2)
        depot_demand = torch.zeros(size=(self.batch_size, self.depot_size if not self.env_params["depot_trick"] else self.env_params["num_virtual_depots"]))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        if (node_demand < 0).any():
            self.attribute_b = True

            self.pomo_size = min(int(self.problem_size * (1 - 0.2)), self.pomo_size)
            if self.env_params["depot_trick"]:
                self.pomo_size += self.env_params["num_virtual_depots"] - 1
            self.start_node = torch.arange(start=1, end=self.problem_size + self.depot_size)[None, :].expand(self.batch_size, -1)
            if self.env_params["depot_trick"]:
                self.start_node = self.start_node[self.depot_node_demand[:, 1:] >= 0].reshape(self.batch_size, -1)[:,
                                  :self.pomo_size]
            else:
                self.start_node = self.start_node[node_demand > 0].reshape(self.batch_size, -1)[:, :self.pomo_size]

            is_backhaul = torch.ones(size=(self.batch_size, self.problem_size))

            if aug_factor > 1:
                is_backhaul.repeat(aug_factor, 1)
            # shape: (batch, problem)
            self.is_backhaul = is_backhaul
        else:
            self.attribute_b = False

            self.pomo_size = self.problem_size + self.depot_size - 1
            if self.env_params["depot_trick"]:
                self.pomo_size = self.problem_size + self.env_params["num_virtual_depots"] - 1

            self.start_node = torch.arange(start=1, end=self.pomo_size + 1)[None, :].expand(self.batch_size, -1)

            is_backhaul = torch.zeros(
                size=(self.batch_size, self.problem_size))
            if aug_factor > 1:
                is_backhaul.repeat(aug_factor, 1)
            # shape: (batch, problem)
            self.is_backhaul = is_backhaul

        if (node_lateTW.sum() > 0):
            self.attribute_tw = True
            depot_servicetime = torch.zeros(size=(self.batch_size, self.depot_size))
            depot_earlyTW = torch.ones(size=(self.batch_size, self.depot_size)) * 0
            depot_lateTW = torch.ones(size=(self.batch_size, self.depot_size)) * 3.0

            is_tw = torch.ones(
                size=(self.batch_size, self.problem_size))
            if aug_factor > 1:
                is_tw.repeat(aug_factor, 1)
            # shape: (batch, problem)
            self.is_tw = is_tw
        else:
            self.attribute_tw = False
            depot_earlyTW = torch.zeros(size=(self.batch_size, self.depot_size))
            # shape: (batch, 1)
            depot_lateTW = torch.zeros(
                size=(self.batch_size,
                      self.depot_size))
            # shape: (batch, 1)
            depot_servicetime = torch.zeros(size=(self.batch_size, self.depot_size))

            is_tw = torch.zeros(
                size=(self.batch_size, self.problem_size))
            if aug_factor > 1:
                is_tw.repeat(aug_factor, 1)
            # shape: (batch, problem)
            self.is_tw = is_tw

        route_limit = route_length_limit.unsqueeze(-1).expand(-1, self.problem_size, -1)

        # Extract a single value per batch (since all 50 are the same)
        route_limit = route_limit[:, 0, 0]  # [256]

        # Expand to (batch, pomo, num_nodes)
        self.route_limit = route_limit.view(self.batch_size, 1, 1).expand(-1, self.pomo_size, self.problem_size)

        # shape: (batch, 1)
        self.depot_node_earlyTW = torch.cat((depot_earlyTW, node_earlyTW), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_lateTW = torch.cat((depot_lateTW, node_lateTW), dim=1)
        # shape: (batch, problem+1)
        self.depot_node_servicetime = torch.cat((depot_servicetime, node_servicetime), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        if (node_demand.sum() > 0):
            self.attribute_c = True
        else:
            self.attribute_c = False
        if (is_open.sum() > 0):
            self.attribute_o = True
        else:
            self.attribute_o = False
        if (route_length_limit.sum() > 0):
            self.attribute_l = True
            is_limit = torch.ones(
                size=(self.batch_size, self.problem_size))
            if aug_factor > 1:
                is_limit.repeat(aug_factor, 1)
            # shape: (batch, problem)
            self.is_limit = is_limit
        else:
            self.attribute_l = False
            is_limit = torch.zeros(
                size=(self.batch_size, self.problem_size))
            if aug_factor > 1:
                is_limit.repeat(aug_factor, 1)
            # shape: (batch, problem)
            self.is_limit = is_limit

        self.reset_state.depot_xy = depot_xy
        self.reset_state.node_xy = node_xy
        self.reset_state.node_demand = node_demand
        self.reset_state.node_earlyTW = node_earlyTW
        self.reset_state.node_lateTW = node_lateTW
        self.reset_state.node_serviceTime = node_servicetime
        self.reset_state.is_open = is_open
        self.reset_state.is_interdepot = is_interdepot
        self.reset_state.is_tw = is_tw
        self.reset_state.is_backhaul = is_backhaul
        self.reset_state.is_limit = is_limit

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX
        self.step_state.start_node = self.start_node

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.departure_depot = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        # shape: (batch_size, pomo)
        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch_size, pomo)
        self.last_at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch_size, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.time = torch.zeros(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.length = torch.zeros(size=(self.batch_size, self.pomo_size))
        # # shape: (batch, pomo)
        if self.attribute_o:
            self.is_open = torch.ones((self.batch_size, self.pomo_size))
        else:
            self.is_open = torch.zeros((self.batch_size, self.pomo_size))
        self.is_interdepot = torch.zeros((self.batch_size, self.pomo_size))

        if self.attribute_id:
            self.visited_ninf_flag = torch.zeros(
                size=(self.batch_size, self.pomo_size, self.problem_size + (self.depot_size * 2)))
            # shape: (batch, pomo, problem+depot*2)
            self.ninf_mask = torch.zeros(
                size=(self.batch_size, self.pomo_size, self.problem_size + (self.depot_size * 2)))
            # shape: (batch, pomo, problem+depot*2)
        else:
            self.visited_ninf_flag = torch.zeros(
                size=(self.batch_size, self.pomo_size, self.problem_size + self.depot_size))
            # shape: (batch, pomo, problem+depot)
            self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size + self.depot_size))
            # shape: (batch, pomo, problem+depot)

        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.last_depot = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.long)
        # shape: (batch, pomo)
        self.current_coord = self.depot_node_xy[:, :1, :]  # depot
        # shape: (batch, pomo, 2)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.time = self.time
        self.step_state.is_open = self.is_open
        self.step_state.is_interdepot = self.is_interdepot
        self.step_state.length = self.length.clone()

        reward = None
        done = False
        return self.step_state, reward, done

    def step(self, selected):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        ####################################
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)

        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        ####################################

        self.at_the_depot = (selected < self.depot_size)
        self.last_depot[self.at_the_depot] = self.current_node[self.at_the_depot]
        self.departure_depot[self.at_the_depot] = self.current_node[self.at_the_depot]

        #### update load information ###

        demand_list = self.depot_node_demand[:, None, :].expand(-1, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill loaded at the depot

        current_coord = self.depot_node_xy[torch.arange(self.batch_size)[:, None], selected]
        self.current_coord = current_coord

        #### mask nodes if load exceed ###

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0:self.depot_size][
            ~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        if self.attribute_b:
            unvisited_demand = demand_list + self.visited_ninf_flag
            # shape: (batch, pomo, problem+1)
            linehauls_unserved = torch.where(unvisited_demand > 0., True, False)
            reset_index = self.at_the_depot & (~linehauls_unserved.any(dim=-1))
            # shape: (batch, pomo)
            self.load[reset_index] = 0.

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.000001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        if self.attribute_b:
            exceed_capacity = self.load[:, :, None] - demand_list > 1.0 + round_error_epsilon
            self.ninf_mask[exceed_capacity] = float('-inf')

        #### update time&distance information ###

        servicetime_list = self.depot_node_servicetime[:, None, :].expand(-1, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        selected_servicetime = servicetime_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        earlyTW_list = self.depot_node_earlyTW[:, None, :].expand(-1, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        selected_earlyTW = earlyTW_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)

        xy_list = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)
        gathering_index = selected[:, :, None, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, 1, 2)
        selected_xy = xy_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo, 2)

        if self.selected_node_list.size()[2] == 1:
            gathering_index_last = self.selected_node_list[:, :, -1][:, :, None, None].expand(-1, -1, -1, 2)
            # shape: (batch, pomo, 1,2)
        else:
            gathering_index_last = self.selected_node_list[:, :, -2][:, :, None, None].expand(-1, -1, -1, 2)
            # shape: (batch, pomo, 1,2)
        last_xy = xy_list.gather(dim=2, index=gathering_index_last).squeeze(dim=2)
        # shape: (batch, pomo, 2)
        selected_time = ((selected_xy - last_xy) ** 2).sum(dim=2).sqrt()  # assumes speed = 1 (time = distance)
        # shape: (batch, pomo)

        self.length += selected_time
        self.length[self.at_the_depot] = 0

        # update time window attribute if it is used
        if (self.attribute_tw):
            if self.attribute_o:
                self.time = torch.max(self.time + selected_time,
                                              self.depot_node_earlyTW[torch.arange(self.batch_size)[:, None], selected]) + \
                                    self.depot_node_servicetime[torch.arange(self.batch_size)[:, None], selected]
                self.time[self.at_the_depot] = 0
                # shape: (batch, pomo)
                arrival_time = torch.max(self.time[:, :, None] + (
                            self.current_coord[:, :, None, :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size,
                                                                                                         -1, -1)).norm(p=2,
                                                                                                                       dim=-1),
                                         self.depot_node_earlyTW[:, None, :].expand(-1, self.pomo_size, -1))
                out_of_tw = arrival_time > self.depot_node_lateTW[:, None, :].expand(-1, self.pomo_size,
                                                                                     -1) + round_error_epsilon
                # shape: (batch, pomo, problem+1)
                out_of_tw[:, :, 0] = False
                self.ninf_mask[out_of_tw] = float('-inf')
            else:
                self.time = torch.max((self.time + selected_time), selected_earlyTW)
                self.time += selected_servicetime
                # shape: (batch, pomo)
                self.time[self.at_the_depot] = 0  # refill time at the depot

                arrival_time = torch.max(self.time[:, :, None] + (
                        self.depot_node_xy[torch.arange(self.batch_size)[:, None], selected][:, :, None,
                        :] - self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1,
                                                                      -1)).norm(p=2,
                                                                                dim=-1),
                                         self.depot_node_earlyTW[:, None, :].expand(-1, self.pomo_size, -1))
                out_of_tw = arrival_time > self.depot_node_lateTW[:, None, :].expand(-1, self.pomo_size,
                                                                                     -1) + round_error_epsilon

                # shape: (batch, pomo, problem+1)
                self.ninf_mask[out_of_tw] = float('-inf')
                selected_depots = self.depot_node_xy[
                    torch.arange(self.depot_node_xy.shape[0]).unsqueeze(1), self.departure_depot]

                # Compute the Euclidean distance using the selected depots
                distance_tensor = (selected_depots[:, :, None, :] - self.depot_node_xy[:, None, :, :]).norm(p=2, dim=-1)
                fail_return_depot = arrival_time + self.depot_node_servicetime[:, None, :].expand(-1, self.pomo_size,
                                                                                                  -1) + distance_tensor > 3.0 + round_error_epsilon

                self.ninf_mask[fail_return_depot] = float('-inf')
                # shape: (batch, pomo, problem+1)

        # update route duration (length) attribute if it is used
        if (self.attribute_l):
            # shape: (batch, pomo)
            depot_xy = xy_list[self.BATCH_IDX, self.POMO_IDX, self.departure_depot, :]

            candidate_nodes_coords = self.depot_node_xy[:, None, self.depot_size:, :]  # expand dims for broadcasting

            dist_cur_to_next = (selected_xy[:, :, None, :] - candidate_nodes_coords).norm(p=2, dim=-1)
            depot_coords_exp = depot_xy[:, :, None, :]  # [batch, pomo, 1, 2]

            # dist from candidate next nodes to depot of that solution
            dist_next_to_depot = (candidate_nodes_coords - depot_coords_exp).norm(p=2,
                                                                                  dim=-1)  # [batch, pomo, problem_size + depot_size]

            # if open attribute is used, the distance return to depot is not counted
            if self.attribute_o:
                total_length = self.length[:, :,
                None] + dist_cur_to_next  # [batch, pomo, problem_size + depot_size]
                length_too_small = total_length > self.route_limit + round_error_epsilon
                # shape: (batch, pomo, problem+depot_size)
            else:
                total_length = self.length[:, :,
                None] + dist_cur_to_next + dist_next_to_depot  # [batch, pomo, problem_size + depot_size]
                length_too_small = total_length > self.route_limit + round_error_epsilon
                # print(self.step_state.length)
                # print(length_to_next + next_to_depot)
                # print("length_too_large",length_too_large)

            self.ninf_mask[:, :, self.depot_size:][length_too_small] = float('-inf')
            # shape: (batch, pomo, problem+depot_size)

        newly_finished = (self.visited_ninf_flag[:, :, self.depot_size:] == float('-inf')).all(dim=2)
        # newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        # self.ninf_mask[:, :, 0][self.finished] = 0
        self.ninf_mask[self.BATCH_IDX[self.finished], self.POMO_IDX[self.finished], self.last_depot[
            self.finished]] = 0  # do not mask depot for finished episode.

        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished
        self.step_state.time = self.time
        self.step_state.length = self.length

        # returning values
        done = self.finished.all()
        if done:
            reward = -self._get_travel_distance()  # note the minus sign!
        else:
            reward = None

        return self.step_state, reward, done

    def _get_travel_distance(self):
        # print(self.selected_node_list[0])
        index_to_gather = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch_size, pomo, selected_list, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch_size, pomo, node, 2)
        seq_ordered = all_xy.gather(dim=2, index=index_to_gather)
        depot_ordered = self.selected_node_list < self.depot_size
        # shape: (batch_size, pomo, selected_list, 2)
        depot_rolled = depot_ordered.roll(dims=2, shifts=-1)
        if self.attribute_o:
            depot_final = depot_rolled
        else:
            depot_final = depot_ordered * depot_rolled
        seq_rolled = seq_ordered.roll(dims=2, shifts=-1)
        segment_lengths = ((seq_ordered - seq_rolled) ** 2).sum(3).sqrt()
        # shape: (batch_size, pomo, selected_list)
        segment_lengths[depot_final] = 0
        travel_distances = segment_lengths.sum(2)
        # shape: (batch_size, pomo)
        return travel_distances

    def get_node_seq(self):

        gathering_index = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        # shape: (batch, pomo, selected_list_length, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        # shape: (batch, pomo, problem+1, 2)

        ordered_seq = all_xy.gather(dim=2, index=gathering_index)
        # shape: (batch, pomo, selected_list_length, 2)

        return gathering_index, ordered_seq

    def load_dataset(self, path, offset=0, num_samples=1000, device='cuda', task=None):
        assert os.path.splitext(path)[1] == ".pkl", "Unsupported file type (.pkl needed)."
        with open(path, 'rb') as f:
            data = pickle.load(f)[offset: offset + num_samples]

        depot_xy, node_xy, node_demand, capacity = [i[0] for i in data], [i[1] for i in data], [i[2] for i in data], [
            i[3] for i in data]
        depot_xy, node_xy, node_demand, capacity = torch.Tensor(depot_xy), torch.Tensor(node_xy), torch.Tensor(
            node_demand), torch.Tensor(capacity)
        node_demand = node_demand / capacity.view(-1, 1)
        if "L" in task:
            route_length_limit_raw = torch.tensor([i[4] for i in data], device=device, dtype=torch.float)
            route_length_limit = route_length_limit_raw.unsqueeze(1).repeat(1, self.problem_size)
        else:
            route_length_limit = torch.zeros(size=(num_samples, self.problem_size), device=device)
        if "TW" in task:
            is_vrpl = True if "L" in task else False
            node_serviceTime = torch.Tensor([i[5 if is_vrpl else 4] for i in data])
            node_earlyTW = torch.Tensor([i[6 if is_vrpl else 5] for i in data])
            node_lateTW = torch.Tensor([i[7 if is_vrpl else 6] for i in data])
        else:
            node_serviceTime = torch.zeros(size=(num_samples, self.problem_size), device=device)
            node_earlyTW = torch.zeros(size=(num_samples, self.problem_size), device=device)
            node_lateTW = torch.zeros(size=(num_samples, self.problem_size), device=device)

        is_open = torch.zeros(size=(num_samples, self.problem_size), device=device)
        is_interdepot = torch.zeros(size=(num_samples, self.problem_size), device=device)

        # Set to ones if problem_type contains 'O' or 'I'
        if "O" in task:
            is_open = torch.ones(size=(num_samples, self.problem_size), device=device)

        data = (depot_xy, node_xy, node_demand, node_earlyTW, node_lateTW, node_serviceTime, is_open,
                is_interdepot, route_length_limit)
        return data
