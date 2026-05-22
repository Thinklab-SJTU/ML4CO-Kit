r"""
PyVRP solver for routing problems.
"""

# Copyright (c) 2024 Thinklab@SJTU
# ML4CO-Kit is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


import pyvrp
import numpy as np
from pyvrp import Model
from pyvrp.stop import MaxRuntime
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE


class PyVRPSolver(SolverBase):
    """PyVRP solver for CVRP-style routing tasks."""

    def __init__(
        self,
        pyvrp_time_limit: float = 1.0,
        pyvrp_seed: int = 1234,
        pyvrp_distance_scale: int = 10000,
        pyvrp_demand_scale: int = 10000,
        pyvrp_time_scale: int = None,
        pyvrp_display: bool = False,
        optimizer: OptimizerBase = None,
    ):
        super(PyVRPSolver, self).__init__(
            solver_type=SOLVER_TYPE.PYVRP, optimizer=optimizer
        )

        # Set Attributes
        self.pyvrp_time_limit = pyvrp_time_limit
        self.pyvrp_seed = pyvrp_seed
        self.pyvrp_distance_scale = pyvrp_distance_scale
        self.pyvrp_demand_scale = pyvrp_demand_scale
        self.pyvrp_time_scale = pyvrp_time_scale or pyvrp_distance_scale
        self.pyvrp_display = pyvrp_display

    def _solve(self, task_data: TaskBase):
        """Solve the task data using PyVRP solver."""
        if task_data.task_type in [
            TASK_TYPE.CVRP,
            TASK_TYPE.VRPL,
            TASK_TYPE.VRPTW,
            TASK_TYPE.OVRP,
        ]:
            return self._solve_cvrp_like(task_data=task_data)
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )

    def _scale_array(self, values: np.ndarray, scale: int, name: str) -> np.ndarray:
        """Scale numeric values to PyVRP integer data."""
        if scale <= 0:
            raise ValueError(f"``{name}`` scale should be positive.")
        return np.rint(values * scale).astype(np.int64)

    def _round_to_int(self, value: float) -> int:
        """Round a numeric value to an integer."""
        return int(np.rint(value))

    def _prepare_common_data(self, task_data: TaskBase):
        """Validate and scale common CVRP-like fields."""
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

        coords = self._scale_array(
            values=np.asarray(task_data.coords, dtype=np.float64),
            scale=self.pyvrp_distance_scale,
            name="distance",
        )
        demands = self._scale_array(
            values=demands,
            scale=self.pyvrp_demand_scale,
            name="demand",
        )
        capacity = self._round_to_int(float(task_data.capacity) * self.pyvrp_demand_scale)
        dists = self._scale_array(
            values=np.asarray(task_data._get_dists(), dtype=np.float64),
            scale=self.pyvrp_distance_scale,
            name="distance",
        )

        if capacity <= 0:
            raise ValueError("Scaled capacity should be positive.")
        if np.any((task_data.demands > 0) & (demands <= 0)):
            raise ValueError(
                "Some positive demands became zero after scaling. Increase "
                "``pyvrp_demand_scale``."
            )

        return coords, demands, capacity, dists

    def _prepare_time_window_data(self, task_data: TaskBase):
        """Validate and scale VRPTW fields."""
        if task_data.task_type != TASK_TYPE.VRPTW:
            return None, None

        task_data._check_time_windows_not_none()
        task_data._check_service_times_not_none()
        time_windows = self._scale_array(
            values=np.asarray(task_data.time_windows, dtype=np.float64),
            scale=self.pyvrp_time_scale,
            name="time",
        )
        service_times = self._scale_array(
            values=np.asarray(task_data.service_times, dtype=np.float64),
            scale=self.pyvrp_time_scale,
            name="time",
        )
        if np.any(time_windows[:, 0] > time_windows[:, 1]):
            raise ValueError("Each time window should satisfy early <= late.")
        return time_windows, service_times

    def _add_depots(self, model: Model, coords: np.ndarray, time_windows: np.ndarray):
        """Add depots to the PyVRP model."""
        depot_kwargs = dict()
        if time_windows is not None:
            depot_kwargs["tw_early"] = int(time_windows[0, 0])
            depot_kwargs["tw_late"] = int(time_windows[0, 1])

        start_depot = model.add_depot(
            x=int(coords[0, 0]), y=int(coords[0, 1]), **depot_kwargs
        )
        return [start_depot]

    def _add_open_route_depots(self, model: Model, coords: np.ndarray):
        """Add start and dummy end depots for OVRP."""
        start_depot = model.add_depot(x=int(coords[0, 0]), y=int(coords[0, 1]))
        end_depot = model.add_depot(x=int(coords[0, 0]), y=int(coords[0, 1]))
        return [start_depot, end_depot]

    def _add_clients(
        self,
        model: Model,
        task_data: TaskBase,
        coords: np.ndarray,
        demands: np.ndarray,
        time_windows: np.ndarray,
        service_times: np.ndarray,
    ):
        """Add clients to the PyVRP model."""
        clients = list()
        for idx in range(task_data.nodes_num):
            kwargs = dict()
            if task_data.task_type == TASK_TYPE.VRPTW:
                kwargs["service_duration"] = int(service_times[idx + 1])
                kwargs["tw_early"] = int(time_windows[idx + 1, 0])
                kwargs["tw_late"] = int(time_windows[idx + 1, 1])
            client = model.add_client(
                x=int(coords[idx + 1, 0]),
                y=int(coords[idx + 1, 1]),
                delivery=int(demands[idx]),
                **kwargs
            )
            clients.append(client)
        return clients

    def _add_vehicle_type(
        self,
        model: Model,
        task_data: TaskBase,
        depots: list,
        capacity: int,
        time_windows: np.ndarray,
    ):
        """Add a vehicle type to the PyVRP model."""
        kwargs = dict(
            num_available=task_data.nodes_num,
            capacity=capacity,
            start_depot=depots[0],
            end_depot=depots[-1],
        )

        if task_data.task_type == TASK_TYPE.VRPL:
            task_data._check_max_route_length_not_none()
            kwargs["max_distance"] = self._round_to_int(
                float(task_data.max_route_length) * self.pyvrp_distance_scale
            )

        if task_data.task_type == TASK_TYPE.VRPTW:
            kwargs["tw_early"] = int(time_windows[0, 0])
            kwargs["tw_late"] = int(time_windows[0, 1])

        model.add_vehicle_type(**kwargs)

    def _open_route_distance(self, dists: np.ndarray, start_idx: int, end_idx: int) -> int:
        """Return distance for the OVRP dummy-end-depot transformation."""
        if start_idx == end_idx:
            return 0
        if end_idx == 1:
            return 0
        if start_idx == 1:
            return int(dists[0, 0 if end_idx == 0 else end_idx - 1])

        start_node = 0 if start_idx == 0 else start_idx - 1
        end_node = 0 if end_idx == 0 else end_idx - 1
        return int(dists[start_node, end_node])

    def _add_edges(
        self,
        model: Model,
        task_data: TaskBase,
        locations: list,
        dists: np.ndarray,
        service_times: np.ndarray,
    ):
        """Add distance and duration edges to the PyVRP model."""
        for start_idx, start in enumerate(locations):
            for end_idx, end in enumerate(locations):
                if task_data.task_type == TASK_TYPE.OVRP:
                    # PyVRP exposes start/end depots rather than an open-route
                    # flag. A dummy zero-cost end depot models the OVRP objective.
                    distance = self._open_route_distance(
                        dists=dists, start_idx=start_idx, end_idx=end_idx
                    )
                    duration = distance
                else:
                    distance = int(dists[start_idx, end_idx])
                    duration = distance
                    if task_data.task_type == TASK_TYPE.VRPTW and start_idx == 0:
                        duration += int(service_times[0])

                model.add_edge(start, end, distance=distance, duration=duration)

    def _routes_to_flat_solution(self, routes: list) -> np.ndarray:
        """Convert route lists to ML4CO-Kit flat CVRP solution."""
        sol = [0]
        for route in routes:
            if len(route) == 0:
                continue
            sol.extend(route)
            sol.append(0)
        return np.array(sol, dtype=np.int32)

    def _extract_routes(self, solution: pyvrp.Solution, num_depots: int) -> list:
        """Extract ML4CO-Kit route lists from a PyVRP solution."""
        routes = list()
        for route in solution.routes():
            visits = list()
            for visit in route.visits():
                if int(visit) < num_depots:
                    continue
                visits.append(int(visit) - num_depots + 1)
            if len(visits) > 0:
                routes.append(visits)
        return routes

    def _solve_cvrp_like(self, task_data: TaskBase):
        """Solve a CVRP-like task and write the solution back."""
        coords, demands, capacity, dists = self._prepare_common_data(task_data=task_data)
        time_windows, service_times = self._prepare_time_window_data(task_data=task_data)

        model = Model()
        if task_data.task_type == TASK_TYPE.OVRP:
            depots = self._add_open_route_depots(model=model, coords=coords)
        else:
            depots = self._add_depots(
                model=model, coords=coords, time_windows=time_windows
            )
        clients = self._add_clients(
            model=model,
            task_data=task_data,
            coords=coords,
            demands=demands,
            time_windows=time_windows,
            service_times=service_times,
        )
        self._add_vehicle_type(
            model=model,
            task_data=task_data,
            depots=depots,
            capacity=capacity,
            time_windows=time_windows,
        )
        self._add_edges(
            model=model,
            task_data=task_data,
            locations=depots + clients,
            dists=dists,
            service_times=service_times,
        )

        result = model.solve(
            stop=MaxRuntime(self.pyvrp_time_limit),
            seed=self.pyvrp_seed,
            display=self.pyvrp_display,
        )
        if not result.is_feasible():
            raise RuntimeError("PyVRP did not find a feasible solution.")

        routes = self._extract_routes(
            solution=result.best,
            num_depots=len(depots),
        )
        sol = self._routes_to_flat_solution(routes=routes)
        task_data.from_data(sol=sol, ref=False)

        if not task_data.check_constraints(task_data.sol):
            raise RuntimeError(
                "PyVRP returned a solution that is invalid in ML4CO-Kit format."
            )

        task_data.evaluate(task_data.sol)
        return task_data
