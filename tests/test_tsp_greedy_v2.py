r"""
Test TSP greedy insertion over candidate edges (``tsp_greedy_v2``).
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


import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, root_folder)


import numpy as np
from ml4co_kit.solver.common.lib.greedy.tsp_greedy import _tsp_greedy_v2


def _is_tour(tour: np.ndarray, nodes_num: int) -> bool:
    return tour[0] == tour[-1] and np.array_equal(np.sort(tour[:-1]), np.arange(nodes_num))


def _knn_candidate_edges(points: np.ndarray, k: int) -> np.ndarray:
    dists = np.linalg.norm(points[:, None] - points[None], axis=-1)
    np.fill_diagonal(dists, np.inf)
    nbrs = np.argsort(dists, axis=1)[:, :k]
    src = np.repeat(np.arange(len(points)), k)
    dst = nbrs.reshape(-1)
    order = np.argsort(dists[src, dst], kind="stable")
    return np.stack([src[order], dst[order]], axis=1).reshape(-1)


def test_tsp_greedy_v2_complete_candidates():
    # Path 0-1-2-3 from the first edges; (0, 2) would close a premature cycle; (3, 0) closes the tour
    edges = np.array([0, 1, 1, 2, 2, 3, 0, 2, 3, 0], dtype=np.int32)
    tour = _tsp_greedy_v2(edges, nodes_num=4)
    assert _is_tour(tour, 4)
    assert {tuple(sorted(e)) for e in zip(tour[:-1], tour[1:])} == {(0, 1), (1, 2), (2, 3), (0, 3)}


def test_tsp_greedy_v2_sparse_candidates():
    rng = np.random.default_rng(1234)
    points = rng.random((500, 2))
    edges = _knn_candidate_edges(points, k=3)

    # kNN candidates alone do not form a tour
    try:
        _tsp_greedy_v2(edges, nodes_num=500)
        raise AssertionError("expected RuntimeError for candidate edges that do not form a tour")
    except RuntimeError:
        pass

    # With coordinates, the remaining fragments are joined by nearest endpoints
    tour = _tsp_greedy_v2(edges, nodes_num=500, points=points)
    assert _is_tour(tour, 500)

    # Batched mode with per-instance and shared coordinates
    batch_points = rng.random((4, 500, 2))
    batch_edges = np.stack([_knn_candidate_edges(p, k=3) for p in batch_points])
    tours = _tsp_greedy_v2(batch_edges, nodes_num=500, num_workers=2, points=batch_points)
    for b in range(4):
        assert _is_tour(tours[b], 500)
        single = _tsp_greedy_v2(batch_edges[b], nodes_num=500, points=batch_points[b])
        assert np.array_equal(tours[b], single)
    tours = _tsp_greedy_v2(batch_edges, nodes_num=500, num_workers=2, points=batch_points[0])
    assert all(_is_tour(t, 500) for t in tours)


def test_tsp_greedy_v2_invalid_inputs():
    for edges, error in [
        (np.array([0, 4], dtype=np.int32), IndexError),
        (np.array([0, 1, 2], dtype=np.int32), ValueError),
    ]:
        try:
            _tsp_greedy_v2(edges, nodes_num=4)
            raise AssertionError(f"expected {error.__name__}")
        except error:
            pass


if __name__ == "__main__":
    test_tsp_greedy_v2_complete_candidates()
    test_tsp_greedy_v2_sparse_candidates()
    test_tsp_greedy_v2_invalid_inputs()
