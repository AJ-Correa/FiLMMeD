import math
import numpy as np
from typing import List

from pyvrp._pyvrp import Route
from pyvrp import Model, SolveParams
from pyvrp.PenaltyManager import PenaltyManager
from pyvrp._pyvrp import (
    RandomNumberGenerator,
    Solution,
)
from pyvrp.search import (
    LocalSearch,
    compute_neighbours,
)
from pyvrp.crossover import selective_route_exchange as srex

def build_solution(nodes: np.ndarray, depot_size: int) -> List[List[int]]:
    """
    Build VRP routes from a sequence of nodes
    """
    routes: List[List[int]] = []
    route: List[int] = []

    for node in nodes:
        node = int(node)
        if node < depot_size and not route:
            route.append(node)
        elif (node < depot_size and route):
            has_non_depot = any(x >= depot_size for x in route)
            if has_non_depot:
                routes.append(route)
            route = [node]
        elif node >= depot_size:
            route.append(node)

    if route and route[-1] >= depot_size:
        routes.append(route)

    return routes


def run_local_search(sol, tester_params, num_depots, customer_coords, demands, depots_coords, capacities, route_length_limits=None,
                     service_durations=None, early_tws=None, late_tws=None):
    """Function to perform local search for a single instance."""
    seed = tester_params['seed']
    num_ls_iters = tester_params['ls_params']['num_ls_iters']
    granular_neighborhood = tester_params['ls_params']['granular_neighborhood']

    search = Search(customer_coords, demands, depots_coords, capacities, route_length_limits, service_durations,
                    early_tws, late_tws, seed=seed, num_iters=num_ls_iters, granular_neighborhood=granular_neighborhood)
    routes = build_solution(sol, num_depots)

    # Run local search optimization on each route
    ls_routes = []
    for r in routes:
        depot = r[0]
        ls_routes.append(Route(search.m.data(), r[1:], depot))

    solution = Solution(search.m.data(), ls_routes)
    search.improve(solution)

    best_sol = []
    distances = []

    for route in search.best_sol.routes():
        best_sol.append([route.start_depot()])
        best_sol[-1].extend(route.visits())
        best_sol[-1].append(route.end_depot())
        distances.append(route.distance_cost())

    # Return the best distance for this instance after local search
    return search.best_dist / search.scale


class Search:
    def __init__(self, customer_coords, demands, depots_coords, capacities,
                 route_length_limits, service_durations, early_tws, late_tws,
                 seed=0, num_iters=10, granular_neighborhood=20):
        self.m = Model()

        self.scale = 1000000

        customers_coords = [[int(x * self.scale) for x in coord] for coord in customer_coords]
        demands = [int(d * self.scale) for d in demands]
        depots_coords = [[int(x * self.scale) for x in coord] for coord in depots_coords]
        capacities = [int(c * self.scale) for c in capacities]
        route_length_limits = (
            [int(r * self.scale) for r in route_length_limits]
            if route_length_limits is not None else None
        )
        service_durations = (
            [int(sd * self.scale) for sd in service_durations]
            if service_durations is not None else None
        )
        early_tws = (
            [int(et * self.scale) for et in early_tws]
            if early_tws is not None else None
        )
        late_tws = (
            [int(lt * self.scale) for lt in late_tws]
            if late_tws is not None else None
        )

        is_tw = (service_durations is not None
                and any(sd != 0 for sd in service_durations))

        num_customers = len(customers_coords)
        num_depots = len(depots_coords)
        self.num_depots = num_depots
        max_vehicles = num_customers

        for idx in range(num_depots):
            depot = self.m.add_depot(x=depots_coords[idx][0], y=depots_coords[idx][1])
            max_distance = route_length_limits[0] if route_length_limits is not None else 0
            if max_distance != 0:
                self.m.add_vehicle_type(max_vehicles, start_depot=depot, end_depot=depot, capacity=[capacities[0]], max_distance=max_distance)
            else:
                self.m.add_vehicle_type(max_vehicles, start_depot=depot, end_depot=depot, capacity=[capacities[0]])

        for idx in range(num_customers):
            if is_tw:
                self.m.add_client(x=customers_coords[idx][0], y=customers_coords[idx][1], delivery=demands[idx],
                                  service_duration=service_durations[idx], tw_early=early_tws[idx], tw_late=late_tws[idx])
            else:
                self.m.add_client(x=customers_coords[idx][0], y=customers_coords[idx][1], delivery=demands[idx])

        for frm_idx, frm in enumerate(self.m.locations):
            for to_idx, to in enumerate(self.m.locations):
                distance = int(math.dist((frm.x, frm.y), (to.x, to.y)))
                if is_tw:
                    self.m.add_edge(frm, to, distance=distance, duration=distance)
                else:
                    self.m.add_edge(frm, to, distance=distance)

        params = SolveParams()
        params.neighbourhood.nb_granular = granular_neighborhood
        params.penalty.repair_booster = 30

        self.rng = RandomNumberGenerator(seed=seed)
        neighbours = compute_neighbours(self.m.data(), params.neighbourhood)
        ls = LocalSearch(self.m.data(), self.rng, neighbours)

        for node_op in params.node_ops:
            ls.add_node_operator(node_op(self.m.data()))

        for route_op in params.route_ops:
            ls.add_route_operator(route_op(self.m.data()))

        self._pm = PenaltyManager.init_from(self.m.data(), params.penalty)
        self.crossover = srex
        self._search = ls
        self.best_sol = None
        self.best_dist = 1000000000000000
        self.num_iters = num_iters

    def improve(self, sol: Solution):
        self.best_dist = sol.distance()
        self.best_sol = sol

        improved_sol = self._search(sol, self._pm.cost_evaluator())
        if (improved_sol.distance() < self.best_dist) and (improved_sol.is_feasible()):
            self.best_dist = improved_sol.distance()
            self.best_sol = improved_sol

        solution = sol
        for _ in range(self.num_iters):
            offspring = self.crossover(
                (self.best_sol, solution.make_random(self.m.data(), self.rng)),
                self.m.data(),
                self._pm.cost_evaluator(),
                self.rng,
            )
            solution = self._search(offspring, self._pm.cost_evaluator())
            dist = solution.distance()

            if not solution.is_feasible():
                solution = self._search(solution, self._pm.booster_cost_evaluator())
                dist = solution.distance()

            if (dist < self.best_dist) and (solution.is_feasible()):
                self.best_sol = solution
                self.best_dist = dist


if __name__ == '__main__':
    print()
