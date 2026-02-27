r"""
Sequential Routing Solver for Global Routing.

References:
    [1] ISPD 2007/2008 Global Routing Contest.
    [2] Pan & Chu, "FastRoute: An Efficient and High-Quality Global Router",
        VLSI Design 2012.
    [3] Wu et al., "NCTU-GR 2.0: Multithreaded Collision-Aware Global Routing
        with Bounded-Length Maze Routing", TCAD 2013.
    [4] Liu et al., "Global Placement with Deep Learning-Enabled Explicit
        Routability Optimization", DATE 2021.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import numpy as np
from typing import List, Tuple
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE


def _route_two_pin_lshaped(
    p1: Tuple[int, int],
    p2: Tuple[int, int],
    h_usage: np.ndarray,
    v_usage: np.ndarray,
    h_capacity: np.ndarray,
    v_capacity: np.ndarray,
) -> List[Tuple[int, int]]:
    """
    Route a two-pin net using L-shaped routing with congestion-aware 
    bend selection. Tries both L-shaped candidates and picks the one 
    with lower maximum overflow.
    """
    x1, y1 = p1
    x2, y2 = p2
    grid_h, grid_w_minus1 = h_usage.shape
    grid_h_minus1, grid_w = v_usage.shape

    def compute_cost_horizontal_first():
        cost = 0.0
        sx, ex = min(x1, x2), max(x1, x2)
        for x in range(sx, ex):
            if y1 < grid_h and x < grid_w_minus1:
                usage = h_usage[y1, x]
                cap = h_capacity[y1, x] if h_capacity is not None else 1e9
                if usage >= cap:
                    cost += (usage - cap + 1)
        sy, ey = min(y1, y2), max(y1, y2)
        for y in range(sy, ey):
            if y < grid_h_minus1 and x2 < grid_w:
                usage = v_usage[y, x2]
                cap = v_capacity[y, x2] if v_capacity is not None else 1e9
                if usage >= cap:
                    cost += (usage - cap + 1)
        return cost

    def compute_cost_vertical_first():
        cost = 0.0
        sy, ey = min(y1, y2), max(y1, y2)
        for y in range(sy, ey):
            if y < grid_h_minus1 and x1 < grid_w:
                usage = v_usage[y, x1]
                cap = v_capacity[y, x1] if v_capacity is not None else 1e9
                if usage >= cap:
                    cost += (usage - cap + 1)
        sx, ex = min(x1, x2), max(x1, x2)
        for x in range(sx, ex):
            if y2 < grid_h and x < grid_w_minus1:
                usage = h_usage[y2, x]
                cap = h_capacity[y2, x] if h_capacity is not None else 1e9
                if usage >= cap:
                    cost += (usage - cap + 1)
        return cost

    def apply_horizontal_first():
        path = [p1]
        sx, ex = min(x1, x2), max(x1, x2)
        step_x = 1 if x2 >= x1 else -1
        cx, cy = x1, y1
        for x in range(sx, ex):
            hx = min(cx, cx + step_x)
            if cy < grid_h and hx < grid_w_minus1:
                h_usage[cy, hx] += 1
            cx += step_x
            path.append((cx, cy))
        step_y = 1 if y2 >= y1 else -1
        sy, ey = min(y1, y2), max(y1, y2)
        for y in range(sy, ey):
            vy = min(cy, cy + step_y)
            if vy < grid_h_minus1 and cx < grid_w:
                v_usage[vy, cx] += 1
            cy += step_y
            path.append((cx, cy))
        return path

    def apply_vertical_first():
        path = [p1]
        cx, cy = x1, y1
        step_y = 1 if y2 >= y1 else -1
        sy, ey = min(y1, y2), max(y1, y2)
        for y in range(sy, ey):
            vy = min(cy, cy + step_y)
            if vy < grid_h_minus1 and cx < grid_w:
                v_usage[vy, cx] += 1
            cy += step_y
            path.append((cx, cy))
        step_x = 1 if x2 >= x1 else -1
        sx, ex = min(x1, x2), max(x1, x2)
        for x in range(sx, ex):
            hx = min(cx, cx + step_x)
            if cy < grid_h and hx < grid_w_minus1:
                h_usage[cy, hx] += 1
            cx += step_x
            path.append((cx, cy))
        return path

    # Pick less congested L-shape
    cost_hf = compute_cost_horizontal_first()
    cost_vf = compute_cost_vertical_first()

    if cost_hf <= cost_vf:
        return apply_horizontal_first()
    else:
        return apply_vertical_first()


def _decompose_net_to_two_pin(pins: List[List[int]]) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """
    Decompose a multi-pin net into two-pin subnets using MST (Prim's algorithm).
    This is the standard RSMT approximation approach used in FastRoute.
    """
    if len(pins) <= 1:
        return []
    if len(pins) == 2:
        return [((pins[0][0], pins[0][1]), (pins[1][0], pins[1][1]))]

    n = len(pins)
    in_tree = [False] * n
    min_dist = [float("inf")] * n
    parent = [-1] * n

    in_tree[0] = True
    for j in range(1, n):
        d = abs(pins[0][0] - pins[j][0]) + abs(pins[0][1] - pins[j][1])
        min_dist[j] = d
        parent[j] = 0

    edges = []
    for _ in range(n - 1):
        u = -1
        for j in range(n):
            if not in_tree[j] and (u == -1 or min_dist[j] < min_dist[u]):
                u = j
        in_tree[u] = True
        edges.append(((pins[parent[u]][0], pins[parent[u]][1]),
                       (pins[u][0], pins[u][1])))
        for j in range(n):
            if not in_tree[j]:
                d = abs(pins[u][0] - pins[j][0]) + abs(pins[u][1] - pins[j][1])
                if d < min_dist[j]:
                    min_dist[j] = d
                    parent[j] = u

    return edges


def _sequential_global_routing(
    task_data: TaskBase,
    rip_up_iterations: int,
) -> TaskBase:
    """
    Sequential global routing with net ordering and rip-up & reroute.

    1. Decompose multi-pin nets into two-pin pairs via MST.
    2. Order nets by HPWL (shorter nets first).
    3. Route each two-pin pair with L-shaped congestion-aware routing.
    4. Perform rip-up & reroute iterations to reduce overflow.
    """
    grid_h = task_data.grid_height
    grid_w = task_data.grid_width

    h_usage = np.zeros((grid_h, grid_w - 1), dtype=task_data.precision)
    v_usage = np.zeros((grid_h - 1, grid_w), dtype=task_data.precision)

    h_cap = task_data.h_capacity
    v_cap = task_data.v_capacity

    # Decompose all nets into two-pin subnets
    all_subnets = []
    for net_idx, net in enumerate(task_data.nets):
        pins = net.get("pins", [])
        if len(pins) < 2:
            continue
        two_pin_pairs = _decompose_net_to_two_pin(pins)
        for pair in two_pin_pairs:
            hpwl = abs(pair[0][0] - pair[1][0]) + abs(pair[0][1] - pair[1][1])
            all_subnets.append((net_idx, pair, hpwl))

    # Sort by HPWL (shorter first)
    all_subnets.sort(key=lambda x: x[2])

    # Store routes for potential rip-up
    net_routes = [None] * len(all_subnets)

    # Initial routing
    for i, (net_idx, (p1, p2), hpwl) in enumerate(all_subnets):
        path = _route_two_pin_lshaped(
            p1, p2, h_usage, v_usage, h_cap, v_cap
        )
        net_routes[i] = (p1, p2, path)

    # Rip-up & reroute iterations
    for rr_iter in range(rip_up_iterations):
        # Find subnets that pass through overflowed edges
        overflow_subnets = []
        for i, (net_idx, (p1, p2), hpwl) in enumerate(all_subnets):
            if net_routes[i] is None:
                continue
            _, _, path = net_routes[i]
            has_overflow = False
            for k in range(len(path) - 1):
                cx, cy = path[k]
                nx, ny = path[k + 1]
                if nx != cx:  # horizontal move
                    hx = min(cx, nx)
                    if cy < grid_h and hx < grid_w - 1:
                        if h_cap is not None and h_usage[cy, hx] > h_cap[cy, hx]:
                            has_overflow = True
                            break
                if ny != cy:  # vertical move
                    vy = min(cy, ny)
                    if vy < grid_h - 1 and cx < grid_w:
                        if v_cap is not None and v_usage[vy, cx] > v_cap[vy, cx]:
                            has_overflow = True
                            break
            if has_overflow:
                overflow_subnets.append(i)

        if len(overflow_subnets) == 0:
            break

        # Rip-up overflowed subnets
        for i in overflow_subnets:
            _, _, path = net_routes[i]
            for k in range(len(path) - 1):
                cx, cy = path[k]
                nx, ny = path[k + 1]
                if nx != cx:
                    hx = min(cx, nx)
                    if cy < grid_h and hx < grid_w - 1:
                        h_usage[cy, hx] -= 1
                if ny != cy:
                    vy = min(cy, ny)
                    if vy < grid_h - 1 and cx < grid_w:
                        v_usage[vy, cx] -= 1

        # Reroute
        for i in overflow_subnets:
            net_idx, (p1, p2), hpwl = all_subnets[i]
            path = _route_two_pin_lshaped(
                p1, p2, h_usage, v_usage, h_cap, v_cap
            )
            net_routes[i] = (p1, p2, path)

    # Build solution array [2, H, W]: channel usage
    sol = np.zeros((2, grid_h, grid_w), dtype=task_data.precision)
    sol[0, :, :grid_w - 1] = h_usage
    sol[1, :grid_h - 1, :] = v_usage

    task_data.sol = sol
    return task_data


class SequentialRoutingSolver(SolverBase):
    """
    Sequential Routing Solver for Global Routing.

    Implements a classical sequential global routing approach:
    net decomposition via MST, L-shaped routing with congestion-aware 
    bend selection, and iterative rip-up & reroute. This follows the 
    methodology of FastRoute (Pan & Chu, VLSI Design 2012) and 
    ISPD 2007/2008 contest algorithms.
    """
    def __init__(
        self,
        rip_up_iterations: int = 5,
        optimizer: OptimizerBase = None,
    ):
        # Super Initialization
        super(SequentialRoutingSolver, self).__init__(
            solver_type=SOLVER_TYPE.SEQUENTIAL_ROUTING, optimizer=optimizer
        )

        # Initialize Attributes
        self.rip_up_iterations = rip_up_iterations

    def _solve(self, task_data: TaskBase):
        if task_data.task_type == TASK_TYPE.GLOBAL_ROUTING:
            return _sequential_global_routing(
                task_data=task_data,
                rip_up_iterations=self.rip_up_iterations,
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )
