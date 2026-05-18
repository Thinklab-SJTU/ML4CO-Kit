r"""
Test CVRP PRL utility helpers.
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

from ml4co_kit.task.routing.cvrp import CVRPTask
from ml4co_kit.task.routing.vrp_base import (
    check_capacity,
    check_visit_once,
    evaluate_distance,
    get_capacity_violation,
    get_route_demands,
    make_prior,
    routes_to_flat,
    sol_to_edges,
    sol_to_heatmap,
    sol_to_successor,
    split_routes,
)


TOY_SOLUTION = [0, 1, 2, 0, 3, 4, 0]
TOY_DEMANDS = np.array([0, 2, 3, 2, 1], dtype=np.float32)


def test_split_routes_and_flat_roundtrip():
    solution = np.array(TOY_SOLUTION, dtype=np.int32)
    original = solution.copy()

    assert split_routes(solution) == [[1, 2], [3, 4]]
    assert np.array_equal(solution, original)
    assert routes_to_flat([[1, 2], [3, 4]]) == TOY_SOLUTION


def test_split_routes_skips_empty_and_accepts_missing_boundary_depots():
    assert split_routes([0, 0, 1, 0, 0, 2, 0]) == [[1], [2]]
    assert split_routes((1, 2, 0, 3, 4)) == [[1, 2], [3, 4]]
    assert split_routes([0]) == []
    assert routes_to_flat([[1], [], [2]]) == [0, 1, 0, 2, 0]


def test_routes_to_flat_rejects_depot_inside_route():
    with pytest.raises(ValueError, match="depot"):
        routes_to_flat([[1, 0, 2]])


def test_sol_to_edges_and_heatmap():
    edges = sol_to_edges(TOY_SOLUTION)
    assert edges == [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)]

    heatmap = sol_to_heatmap(TOY_SOLUTION, num_nodes=5)
    assert heatmap.shape == (5, 5)
    for start, end in edges:
        assert heatmap[start, end] == 1
    assert heatmap[1, 3] == 0


def test_check_visit_once_valid_duplicate_missing_and_out_of_range():
    assert check_visit_once(TOY_SOLUTION, num_customers=4) is True
    assert check_visit_once([0, 1, 2, 0, 2, 3, 4, 0], num_customers=4) is False
    assert check_visit_once([0, 1, 2, 0, 3, 0], num_customers=4) is False
    assert check_visit_once([0, 1, 2, 0, 3, 5, 0], num_customers=4) is False


def test_capacity_and_route_demands():
    np.testing.assert_allclose(
        get_route_demands(TOY_SOLUTION, TOY_DEMANDS),
        np.array([5, 3], dtype=np.float64),
    )
    assert check_capacity(TOY_SOLUTION, TOY_DEMANDS, capacity=5) is True
    assert check_capacity(TOY_SOLUTION, TOY_DEMANDS, capacity=4) is False
    np.testing.assert_allclose(
        get_capacity_violation(TOY_SOLUTION, TOY_DEMANDS, capacity=4),
        np.array([1, 0], dtype=np.float64),
    )


def test_sol_to_successor_handles_multiple_depot_successors():
    successor_info = sol_to_successor(TOY_SOLUTION, num_nodes=5)
    successor = successor_info["successor"]

    assert successor_info["depot_successors"] == [1, 3]
    assert successor[0] == -1
    assert successor[1] == 2
    assert successor[2] == 0
    assert successor[3] == 4
    assert successor[4] == 0


def test_make_prior_contains_prl_fields():
    prior = make_prior(TOY_SOLUTION, num_nodes=5)

    assert set(prior.keys()) == {
        "heatmap",
        "edge_list",
        "successor",
        "depot_successors",
        "confidence",
    }
    assert prior["heatmap"].shape == (5, 5)
    assert prior["edge_list"] == sol_to_edges(TOY_SOLUTION)
    assert prior["depot_successors"] == [1, 3]
    np.testing.assert_array_equal(prior["confidence"], np.ones(5, dtype=np.float32))


def test_make_prior_accepts_scalar_and_array_confidence():
    scalar_prior = make_prior(TOY_SOLUTION, num_nodes=5, confidence=0.5)
    np.testing.assert_array_equal(
        scalar_prior["confidence"],
        np.full(5, 0.5, dtype=np.float32),
    )

    confidence = np.array([1.0, 0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    array_prior = make_prior(TOY_SOLUTION, num_nodes=5, confidence=confidence)
    np.testing.assert_array_equal(array_prior["confidence"], confidence)


def test_evaluate_distance_with_matrix_and_coords():
    coords = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [0.0, 1.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    diff = np.expand_dims(coords, 0) - np.expand_dims(coords, 1)
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    from_matrix = evaluate_distance(TOY_SOLUTION, dist_matrix=dist_matrix)
    from_coords = evaluate_distance(TOY_SOLUTION, coords=coords)

    assert np.isclose(from_matrix, from_coords)


def test_cvrp_task_prl_wrappers():
    task = CVRPTask()
    task.from_data(
        depots=np.array([0.0, 0.0], dtype=np.float32),
        points=np.array(
            [
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.0, 2.0],
            ],
            dtype=np.float32,
        ),
        demands=np.array([2, 3, 2, 1], dtype=np.float32),
        capacity=5,
        sol=np.array(TOY_SOLUTION, dtype=np.int32),
    )

    assert task.split_routes() == [[1, 2], [3, 4]]
    assert task.routes_to_flat([[1, 2], [3, 4]]) == TOY_SOLUTION
    assert task.sol_to_edges() == sol_to_edges(TOY_SOLUTION)
    assert task.sol_to_heatmap().shape == (5, 5)
    assert task.check_visit_once() is True
    assert task.check_capacity() is True
    np.testing.assert_allclose(task.get_route_demands(), np.array([5, 3]))
    assert np.isclose(task.evaluate_distance(), task.evaluate(task.sol))

    prior = task.make_prior()
    assert prior["heatmap"].shape == (5, 5)
    assert prior["depot_successors"] == [1, 3]
