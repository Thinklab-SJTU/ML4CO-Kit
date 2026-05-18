r"""
Base utilities for Vehicle Routing Problem solutions.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union


ArrayLike = Union[Sequence[int], np.ndarray]
RoutesLike = Sequence[Sequence[int]]


def _as_1d_int_array(values: ArrayLike, name: str) -> np.ndarray:
    """Convert values to a one-dimensional integer array."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"``{name}`` should be a 1D array.")

    try:
        int_arr = arr.astype(np.int64, copy=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"``{name}`` should contain integer node ids.") from exc

    if not np.array_equal(arr, int_arr):
        raise ValueError(f"``{name}`` should contain integer node ids.")
    return int_arr


def _as_1d_float_array(values: Sequence[float], name: str) -> np.ndarray:
    """Convert values to a one-dimensional float array."""
    arr = np.asarray(values)
    if arr.ndim != 1:
        raise ValueError(f"``{name}`` should be a 1D array.")
    return arr.astype(np.float64, copy=False)


def _validate_num_nodes(num_nodes: int) -> int:
    """Validate and return the number of nodes."""
    num_nodes = int(num_nodes)
    if num_nodes <= 0:
        raise ValueError("``num_nodes`` should be positive.")
    return num_nodes


def _validate_node(node: int, num_nodes: int, name: str = "node"):
    """Validate that a node id is inside [0, num_nodes)."""
    if node < 0 or node >= num_nodes:
        raise ValueError(
            f"``{name}`` id {node} is outside the valid range [0, {num_nodes})."
        )


def split_routes(solution: ArrayLike, depot: int = 0) -> List[List[int]]:
    """
    Split a depot-delimited flat VRP solution into route lists.

    The standard CVRP format is ``[0, ..., 0, ..., 0]`` with depot ``0``.
    Consecutive depots are treated as empty routes and skipped. If the input
    does not start or end with the depot, the remaining non-depot segment is
    still returned as a route for compatibility with partially normalized data.
    """
    sol = _as_1d_int_array(solution, "solution")
    depot = int(depot)

    routes = list()
    route = list()
    for node in sol:
        node = int(node)
        if node == depot:
            if len(route) > 0:
                routes.append(route)
                route = list()
        else:
            route.append(node)

    if len(route) > 0:
        routes.append(route)
    return routes


def routes_to_flat(routes: RoutesLike, depot: int = 0) -> List[int]:
    """
    Convert route lists into a depot-delimited flat VRP solution.

    Routes should not contain the depot. Empty routes are skipped. If no
    non-empty routes are provided, the result is ``[depot]``.
    """
    depot = int(depot)
    flat = [depot]

    for route in routes:
        route_arr = _as_1d_int_array(route, "route")
        if route_arr.size == 0:
            continue
        if np.any(route_arr == depot):
            raise ValueError("Routes should not contain the depot.")
        flat.extend(route_arr.astype(int).tolist())
        flat.append(depot)

    return flat


def sol_to_edges(
    solution: ArrayLike,
    depot: int = 0,
    include_depot_edges: bool = True,
) -> List[Tuple[int, int]]:
    """
    Convert a flat VRP solution into a directed edge list.

    Depot leaving and returning edges are included by default. Future open-route
    VRP variants can build on ``include_depot_edges`` without changing the CVRP
    default behavior.
    """
    edges = list()
    for route in split_routes(solution=solution, depot=depot):
        if include_depot_edges:
            nodes = [int(depot)] + route + [int(depot)]
        else:
            nodes = route
        for idx in range(len(nodes) - 1):
            edges.append((int(nodes[idx]), int(nodes[idx + 1])))
    return edges


def sol_to_heatmap(
    solution: ArrayLike,
    num_nodes: int,
    depot: int = 0,
    include_depot_edges: bool = True,
    dtype: Any = np.float32,
) -> np.ndarray:
    """
    Convert a flat VRP solution into a directed ``num_nodes x num_nodes`` heatmap.

    Entry ``heatmap[i, j]`` is set to ``1`` when directed edge ``(i, j)`` appears
    in the solution. Multiple depot successors are valid for CVRP and share the
    same depot row.
    """
    num_nodes = _validate_num_nodes(num_nodes)
    _validate_node(int(depot), num_nodes, "depot")

    heatmap = np.zeros((num_nodes, num_nodes), dtype=dtype)
    for start, end in sol_to_edges(
        solution=solution,
        depot=depot,
        include_depot_edges=include_depot_edges,
    ):
        _validate_node(start, num_nodes, "edge start")
        _validate_node(end, num_nodes, "edge end")
        heatmap[start, end] = 1
    return heatmap


def sol_to_successor(
    solution: ArrayLike,
    num_nodes: int,
    depot: int = 0,
    missing_value: int = -1,
) -> Dict[str, Union[np.ndarray, List[int]]]:
    """
    Convert a flat VRP solution into customer successors plus depot successors.

    In CVRP the depot can have multiple successors, so ``successor[depot]`` is
    set to ``missing_value`` and all depot successors are returned separately.
    Customer nodes are expected to have at most one successor.
    """
    num_nodes = _validate_num_nodes(num_nodes)
    depot = int(depot)
    _validate_node(depot, num_nodes, "depot")

    successor = np.full(num_nodes, int(missing_value), dtype=np.int64)
    depot_successors = list()
    seen_customers = set()

    for route in split_routes(solution=solution, depot=depot):
        if len(route) == 0:
            continue

        first_node = int(route[0])
        _validate_node(first_node, num_nodes, "depot successor")
        depot_successors.append(first_node)

        for idx, node in enumerate(route):
            node = int(node)
            _validate_node(node, num_nodes, "route node")
            if node == depot:
                raise ValueError("Routes should not contain the depot.")
            if node in seen_customers:
                raise ValueError(
                    f"Customer node {node} appears more than once and cannot "
                    "be represented by a unique successor."
                )

            next_node = int(route[idx + 1]) if idx + 1 < len(route) else depot
            _validate_node(next_node, num_nodes, "successor")
            successor[node] = next_node
            seen_customers.add(node)

    return {
        "successor": successor,
        "depot_successors": depot_successors,
    }


def check_visit_once(
    solution: ArrayLike,
    num_customers: int,
    depot: int = 0,
) -> bool:
    """
    Check whether customers ``1..num_customers`` are visited exactly once.

    Depot occurrences are ignored. Out-of-range customer nodes return ``False``.
    """
    num_customers = int(num_customers)
    if num_customers < 0:
        raise ValueError("``num_customers`` should be non-negative.")

    try:
        sol = _as_1d_int_array(solution, "solution")
    except ValueError:
        return False

    visits = list()
    for node in sol:
        node = int(node)
        if node == depot:
            continue
        if node < 1 or node > num_customers:
            return False
        visits.append(node)

    if len(visits) != num_customers:
        return False
    return len(set(visits)) == num_customers


def get_route_demands(
    solution: ArrayLike,
    demands: Sequence[float],
    depot: int = 0,
) -> np.ndarray:
    """
    Return the demand sum for each route.

    ``demands`` is indexed by node id and should include the depot demand at
    ``demands[depot]``.
    """
    demand_arr = _as_1d_float_array(demands, "demands")
    if demand_arr.size == 0:
        raise ValueError("``demands`` should not be empty.")

    route_demands = list()
    for route in split_routes(solution=solution, depot=depot):
        demand = 0.0
        for node in route:
            node = int(node)
            _validate_node(node, demand_arr.size, "route node")
            demand += float(demand_arr[node])
        route_demands.append(demand)

    return np.asarray(route_demands, dtype=demand_arr.dtype)


def get_capacity_violation(
    solution: ArrayLike,
    demands: Sequence[float],
    capacity: float,
    depot: int = 0,
) -> np.ndarray:
    """
    Return per-route capacity violation amounts.

    Each value is ``max(route_demand - capacity, 0)``.
    """
    capacity = float(capacity)
    route_demands = get_route_demands(solution=solution, demands=demands, depot=depot)
    violations = route_demands - capacity
    violations[violations < 0] = 0
    return violations


def check_capacity(
    solution: ArrayLike,
    demands: Sequence[float],
    capacity: float,
    depot: int = 0,
    threshold: float = 1e-8,
) -> bool:
    """
    Check whether every route demand is within vehicle capacity.

    ``demands`` is indexed by node id and should include the depot demand.
    """
    violations = get_capacity_violation(
        solution=solution,
        demands=demands,
        capacity=capacity,
        depot=depot,
    )
    return bool(np.all(violations <= float(threshold)))


def evaluate_distance(
    solution: ArrayLike,
    dist_matrix: Optional[np.ndarray] = None,
    coords: Optional[np.ndarray] = None,
    depot: int = 0,
    dtype: Any = np.float32,
) -> np.floating:
    """
    Evaluate route distance from a distance matrix or Euclidean coordinates.

    ``dist_matrix`` is preferred when provided. If it is ``None``, Euclidean
    distances are computed from ``coords``.
    """
    if dist_matrix is None and coords is None:
        raise ValueError("Either ``dist_matrix`` or ``coords`` should be provided.")

    result_dtype = np.dtype(dtype).type
    total_distance = result_dtype(0.0)
    edges = sol_to_edges(solution=solution, depot=depot, include_depot_edges=True)

    if dist_matrix is not None:
        matrix = np.asarray(dist_matrix)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            raise ValueError("``dist_matrix`` should be a square 2D array.")
        num_nodes = matrix.shape[0]
        for start, end in edges:
            _validate_node(start, num_nodes, "edge start")
            _validate_node(end, num_nodes, "edge end")
            total_distance += result_dtype(matrix[start, end])
        return total_distance

    coords_arr = np.asarray(coords)
    if coords_arr.ndim != 2:
        raise ValueError("``coords`` should be a 2D array.")
    num_nodes = coords_arr.shape[0]
    for start, end in edges:
        _validate_node(start, num_nodes, "edge start")
        _validate_node(end, num_nodes, "edge end")
        cost = np.linalg.norm(coords_arr[start] - coords_arr[end])
        total_distance += result_dtype(cost)
    return total_distance


def _prepare_confidence(
    confidence: Optional[Union[float, Sequence[float], np.ndarray]],
    num_nodes: int,
    dtype: Any,
) -> np.ndarray:
    """Prepare node-level confidence values."""
    if confidence is None:
        return np.ones(num_nodes, dtype=dtype)
    if np.isscalar(confidence):
        return np.full(num_nodes, float(confidence), dtype=dtype)

    confidence_arr = np.asarray(confidence, dtype=dtype)
    if confidence_arr.ndim != 1 or confidence_arr.shape[0] != num_nodes:
        raise ValueError(
            "``confidence`` should be a scalar or a 1D array with shape "
            f"({num_nodes},)."
        )
    return confidence_arr


def make_prior(
    solution: ArrayLike,
    num_nodes: int,
    depot: int = 0,
    confidence: Optional[Union[float, Sequence[float], np.ndarray]] = None,
    dtype: Any = np.float32,
) -> Dict[str, Any]:
    """
    Build PRL prior data from a flat VRP solution.

    The returned dictionary contains a directed heatmap, edge list, customer
    successor vector, depot successors, and node-level confidence values.
    """
    num_nodes = _validate_num_nodes(num_nodes)
    edge_list = sol_to_edges(solution=solution, depot=depot, include_depot_edges=True)
    heatmap = sol_to_heatmap(
        solution=solution,
        num_nodes=num_nodes,
        depot=depot,
        include_depot_edges=True,
        dtype=dtype,
    )
    successor_info = sol_to_successor(
        solution=solution,
        num_nodes=num_nodes,
        depot=depot,
    )
    confidence_arr = _prepare_confidence(
        confidence=confidence,
        num_nodes=num_nodes,
        dtype=dtype,
    )

    return {
        "heatmap": heatmap,
        "edge_list": edge_list,
        "successor": successor_info["successor"],
        "depot_successors": successor_info["depot_successors"],
        "confidence": confidence_arr,
    }


__all__ = [
    "split_routes",
    "routes_to_flat",
    "sol_to_edges",
    "sol_to_heatmap",
    "sol_to_successor",
    "check_visit_once",
    "get_route_demands",
    "get_capacity_violation",
    "check_capacity",
    "evaluate_distance",
    "make_prior",
]
