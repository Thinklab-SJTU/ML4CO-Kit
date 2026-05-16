r"""
PyVRP backend for CVRP.
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
from typing import Any, Dict, List, Sequence, Tuple
from ml4co_kit.task.routing.cvrp import CVRPTask


def _import_pyvrp():
    """Import PyVRP lazily."""
    try:
        from pyvrp import Model
        from pyvrp.stop import MaxRuntime
    except ImportError as exc:
        raise ImportError(
            "PyVRP is required for PyVRPSolver. Please install it with "
            "`pip install pyvrp`."
        ) from exc
    return Model, MaxRuntime


def _round_to_int(value: float) -> int:
    """Round a numeric value to a Python int."""
    return int(np.rint(value))


def _scale_array(values: np.ndarray, scale: int, name: str) -> np.ndarray:
    """Scale an array to int64 values."""
    if scale <= 0:
        raise ValueError(f"``{name}`` scale should be positive.")
    return np.rint(values * scale).astype(np.int64)


def _add_depot(model: Any, x: int, y: int) -> Any:
    """Add a depot to a PyVRP model across supported APIs."""
    if hasattr(model, "add_location"):
        try:
            location = model.add_location(x=x, y=y)
            return model.add_depot(location=location)
        except TypeError:
            pass

    try:
        return model.add_depot(x=x, y=y)
    except TypeError:
        return model.add_depot(x, y)


def _add_client(model: Any, x: int, y: int, delivery: int) -> Any:
    """Add a client to a PyVRP model across supported APIs."""
    if hasattr(model, "add_location"):
        try:
            location = model.add_location(x=x, y=y)
            return model.add_client(location=location, delivery=delivery)
        except TypeError:
            pass

    try:
        return model.add_client(x=x, y=y, delivery=delivery)
    except TypeError:
        return model.add_client(x, y, delivery)


def _add_vehicle_type(model: Any, num_available: int, capacity: int):
    """Add a vehicle type to a PyVRP model across supported APIs."""
    try:
        return model.add_vehicle_type(
            num_available=num_available,
            capacity=capacity,
        )
    except TypeError:
        return model.add_vehicle_type(capacity, num_available)


def _solve_model(model: Any, MaxRuntime: Any, time_limit: float, seed: int, display: bool):
    """Solve a PyVRP model across supported APIs."""
    stop = MaxRuntime(time_limit)
    try:
        return model.solve(stop=stop, seed=seed, display=display)
    except TypeError:
        try:
            return model.solve(stop=stop, seed=seed)
        except TypeError:
            return model.solve(stop, seed)


def _get_result_best(result: Any) -> Any:
    """Return the best solution from a PyVRP result."""
    best = getattr(result, "best", None)
    if best is None:
        return None
    return best() if callable(best) else best


def _result_is_feasible(result: Any) -> bool:
    """Return feasibility if the PyVRP result exposes it."""
    is_feasible = getattr(result, "is_feasible", None)
    if is_feasible is None:
        return True
    return bool(is_feasible() if callable(is_feasible) else is_feasible)


def _get_solution_routes(best_solution: Any) -> Sequence[Any]:
    """Return routes from a PyVRP solution across supported APIs."""
    if hasattr(best_solution, "routes"):
        routes = getattr(best_solution, "routes")
        return routes() if callable(routes) else routes
    if hasattr(best_solution, "get_routes"):
        routes = getattr(best_solution, "get_routes")
        return routes() if callable(routes) else routes
    raise RuntimeError("Unable to extract routes from the PyVRP solution.")


def _coerce_visit(visit: Any, client_id_map: Dict[int, int]) -> int:
    """Convert a PyVRP route visit to an integer node id."""
    if isinstance(visit, (int, np.integer)):
        return int(visit)

    mapped_visit = client_id_map.get(id(visit))
    if mapped_visit is not None:
        return mapped_visit

    for attr_name in ["idx", "index", "client", "location"]:
        if hasattr(visit, attr_name):
            value = getattr(visit, attr_name)
            value = value() if callable(value) else value
            if isinstance(value, (int, np.integer)):
                return int(value)

    raise RuntimeError(f"Unable to parse PyVRP route visit {visit!r}.")


def _get_route_visits(route: Any, client_id_map: Dict[int, int]) -> List[int]:
    """Return visits from a PyVRP route across supported APIs."""
    if isinstance(route, (list, tuple, np.ndarray)):
        raw_visits = route
    elif hasattr(route, "visits"):
        visits = getattr(route, "visits")
        raw_visits = visits() if callable(visits) else visits
    else:
        raise RuntimeError(f"Unable to parse PyVRP route {route!r}.")

    return [_coerce_visit(visit, client_id_map) for visit in raw_visits]


def _normalise_route_indices(routes: List[List[int]], num_customers: int) -> List[List[int]]:
    """Convert PyVRP route indices to ML4CO-Kit customer ids."""
    visits = [node for route in routes for node in route]
    if len(visits) == 0:
        return routes

    sorted_visits = sorted(visits)
    one_based = list(range(1, num_customers + 1))
    zero_based = list(range(num_customers))
    if sorted_visits == one_based:
        return routes
    if sorted_visits == zero_based:
        return [[node + 1 for node in route] for route in routes]

    raise RuntimeError(
        "Unable to map PyVRP route visits to ML4CO-Kit customer ids. "
        f"Expected {one_based} or {zero_based}, got {sorted_visits}."
    )


def _extract_routes_from_pyvrp_solution(
    best_solution: Any,
    client_id_map: Dict[int, int],
    num_customers: int,
) -> List[List[int]]:
    """Extract ML4CO-Kit route lists from a PyVRP solution."""
    routes = list()
    for route in _get_solution_routes(best_solution):
        visits = _get_route_visits(route, client_id_map)
        if len(visits) > 0:
            routes.append(visits)

    return _normalise_route_indices(routes=routes, num_customers=num_customers)


def _routes_to_flat_solution(routes: List[List[int]]) -> np.ndarray:
    """Convert route lists to ML4CO-Kit flat CVRP solution."""
    flat_sol = [0]
    for route in routes:
        if len(route) == 0:
            continue
        flat_sol.extend(route)
        flat_sol.append(0)
    return np.asarray(flat_sol, dtype=np.int32)


def _prepare_cvrp_data(
    task_data: CVRPTask,
    distance_scale: int,
    demand_scale: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Validate and scale CVRP task data for PyVRP."""
    task_data._check_depots_not_none()
    task_data._check_points_not_none()
    task_data._check_coords_not_none()
    task_data._check_demands_not_none()
    task_data._check_capacity_not_none()

    if task_data.nodes_num is None:
        raise ValueError("``nodes_num`` cannot be None!")

    demands = np.asarray(task_data.demands, dtype=np.float64)
    capacity = float(task_data.capacity)
    if capacity <= 0:
        raise ValueError("``capacity`` should be positive.")
    if np.any(demands > capacity + task_data.threshold):
        raise ValueError("Every customer demand should be no larger than capacity.")

    coords_scaled = _scale_array(
        values=np.asarray(task_data.coords, dtype=np.float64),
        scale=distance_scale,
        name="distance",
    )
    demands_scaled = _scale_array(
        values=demands,
        scale=demand_scale,
        name="demand",
    )
    capacity_scaled = _round_to_int(capacity * demand_scale)
    if capacity_scaled <= 0:
        raise ValueError("Scaled capacity should be positive.")
    if np.any((demands > 0) & (demands_scaled <= 0)):
        raise ValueError(
            "Some positive demands became zero after scaling. Increase "
            "``pyvrp_demand_scale``."
        )

    dists = task_data._get_dists()
    dists_scaled = _scale_array(
        values=np.asarray(dists, dtype=np.float64),
        scale=distance_scale,
        name="distance",
    )
    return coords_scaled, demands_scaled, dists_scaled, capacity_scaled


def cvrp_pyvrp(
    task_data: CVRPTask,
    time_limit: float = 1.0,
    seed: int = 1234,
    distance_scale: int = 10000,
    demand_scale: int = 10000,
    display: bool = False,
):
    """
    Solve a CVRP task with PyVRP and write the solution back to ``task_data``.
    """
    Model, MaxRuntime = _import_pyvrp()

    coords, demands, dists, capacity = _prepare_cvrp_data(
        task_data=task_data,
        distance_scale=distance_scale,
        demand_scale=demand_scale,
    )

    num_customers = task_data.nodes_num
    model = Model()
    depot = _add_depot(model=model, x=int(coords[0, 0]), y=int(coords[0, 1]))
    _add_vehicle_type(
        model=model,
        num_available=num_customers,
        capacity=capacity,
    )

    client_id_map = dict()
    clients = list()
    for idx in range(num_customers):
        client = _add_client(
            model=model,
            x=int(coords[idx + 1, 0]),
            y=int(coords[idx + 1, 1]),
            delivery=int(demands[idx]),
        )
        clients.append(client)
        client_id_map[id(client)] = idx + 1

    locations = [depot] + clients
    for start_idx, start in enumerate(locations):
        for end_idx, end in enumerate(locations):
            model.add_edge(
                start,
                end,
                distance=int(dists[start_idx, end_idx]),
            )

    result = _solve_model(
        model=model,
        MaxRuntime=MaxRuntime,
        time_limit=time_limit,
        seed=seed,
        display=display,
    )
    if not _result_is_feasible(result):
        raise RuntimeError("PyVRP did not find a feasible solution.")

    best_solution = _get_result_best(result)
    if best_solution is None:
        raise RuntimeError("PyVRP did not return a best solution.")

    routes = _extract_routes_from_pyvrp_solution(
        best_solution=best_solution,
        client_id_map=client_id_map,
        num_customers=num_customers,
    )
    sol = _routes_to_flat_solution(routes=routes)
    task_data.from_data(sol=sol, ref=False)

    if not task_data.check_constraints(task_data.sol):
        raise RuntimeError(
            "PyVRP returned a solution that is invalid in ML4CO-Kit format."
        )

    # Run the task evaluator once to ensure the cost can be recomputed by Kit.
    task_data.evaluate(task_data.sol)
