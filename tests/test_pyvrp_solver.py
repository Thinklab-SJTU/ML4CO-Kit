r"""
Test PyVRP solver.
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
import pytest

from ml4co_kit import PyVRPSolver as TopLevelPyVRPSolver
from ml4co_kit.solver import PyVRPSolver
from ml4co_kit.solver.base import SOLVER_TYPE
from ml4co_kit.task.routing.cvrp import CVRPTask
from ml4co_kit.task.routing.tsp import TSPTask


def _build_toy_cvrp_task() -> CVRPTask:
    task = CVRPTask()
    task.from_data(
        depots=np.array([0.0, 0.0], dtype=np.float32),
        points=np.array(
            [
                [1.0, 0.0],
                [2.0, 0.0],
                [0.0, 1.0],
                [0.0, 2.0],
            ],
            dtype=np.float32,
        ),
        demands=np.array([2, 3, 2, 1], dtype=np.float32),
        capacity=5,
    )
    return task


def test_pyvrp_solver_exports_without_importing_pyvrp_backend():
    solver = PyVRPSolver(pyvrp_time_limit=0.1)

    assert solver.solver_type == SOLVER_TYPE.PYVRP
    assert TopLevelPyVRPSolver is PyVRPSolver


def test_pyvrp_solver_rejects_unsupported_task():
    task = TSPTask()
    task.from_data(points=np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32))
    solver = PyVRPSolver(pyvrp_time_limit=0.1)

    with pytest.raises(ValueError, match="only supports CVRP"):
        solver.solve(task)


def test_pyvrp_solver_solves_toy_cvrp():
    pytest.importorskip("pyvrp")

    task = _build_toy_cvrp_task()
    solver = PyVRPSolver(pyvrp_time_limit=1.0, pyvrp_seed=1234)
    solver.solve(task)

    assert task.sol is not None
    assert task.sol[0] == 0
    assert task.sol[-1] == 0
    assert task.check_constraints(task.sol) is True
    assert task.check_visit_once(task.sol) is True

    cost = task.evaluate(task.sol)
    assert np.isfinite(cost)
    assert cost > 0

    prior = task.make_prior(task.sol)
    assert prior["heatmap"].shape == (task.nodes_num + 1, task.nodes_num + 1)
    assert len(prior["edge_list"]) > 0
    assert prior["successor"].shape == (task.nodes_num + 1,)
    assert prior["confidence"].shape == (task.nodes_num + 1,)
