r"""
Force-Directed Solver for Standard Cell Placement.

References:
    [1] Eisenmann & Johannes, "Generic global placement and floorplanning",
        DAC 1998.
    [2] Viswanathan & Chu, "FastPlace: Efficient analytical placement using cell 
        shifting, iterative local refinement, and a hybrid net model", 
        ISPD 2004.
    [3] Lin et al., "DREAMPlace: Deep Learning Toolkit-Enabled GPU Acceleration 
        for Modern VLSI Placement", DAC 2019.
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
from ml4co_kit.optimizer.base import OptimizerBase
from ml4co_kit.task.base import TaskBase, TASK_TYPE
from ml4co_kit.solver.base import SolverBase, SOLVER_TYPE


def _force_directed_std_cell_placement(
    task_data: TaskBase,
    max_iter: int,
    learning_rate: float,
    density_weight: float,
    grid_size: int,
) -> TaskBase:
    """
    Force-directed placement for standard cells.

    Uses a spring model for wirelength (star net model with HPWL gradient 
    approximation) combined with a density force that spreads cells away 
    from over-dense regions. This mirrors the core loop of analytical 
    placement engines like FastPlace and ePlace/DREAMPlace.
    """
    n = task_data.std_cells_num
    canvas_w = task_data.canvas_width
    canvas_h = task_data.canvas_height

    # Initialize: random placement within canvas
    pos = np.zeros((n, 2), dtype=task_data.precision)
    for i in range(n):
        w_i = task_data.std_cells[i]["width"]
        h_i = task_data.std_cells[i]["height"]
        pos[i, 0] = np.random.uniform(w_i / 2, canvas_w - w_i / 2)
        pos[i, 1] = np.random.uniform(h_i / 2, canvas_h - h_i / 2)

    grid_w = canvas_w / grid_size
    grid_h = canvas_h / grid_size

    for iteration in range(max_iter):
        force = np.zeros_like(pos)

        # --- Wirelength force (star net model) ---
        for net in task_data.nets:
            cell_indices = net.get("cells", [])
            if len(cell_indices) < 2:
                continue

            valid_indices = [idx for idx in cell_indices if idx < n]
            if len(valid_indices) < 2:
                continue

            pin_pos = pos[valid_indices]
            centroid = pin_pos.mean(axis=0)

            for idx in valid_indices:
                # Spring force toward net centroid (B2B net model approximation)
                diff = centroid - pos[idx]
                force[idx] += diff

        # --- Density force ---
        density_map = np.zeros((grid_size, grid_size), dtype=task_data.precision)
        cell_grid = np.zeros((n, 2), dtype=np.int32)  # grid index for each cell

        for i in range(n):
            gx = int(np.clip(pos[i, 0] / grid_w, 0, grid_size - 1))
            gy = int(np.clip(pos[i, 1] / grid_h, 0, grid_size - 1))
            cell_grid[i] = [gx, gy]
            area = task_data.std_cells[i]["width"] * task_data.std_cells[i]["height"]
            density_map[gy, gx] += area

        # Account for fixed macros
        for macro in task_data.fixed_macros:
            mx, my = macro["x"], macro["y"]
            mw, mh = macro["width"], macro["height"]
            gx = int(np.clip(mx / grid_w, 0, grid_size - 1))
            gy = int(np.clip(my / grid_h, 0, grid_size - 1))
            density_map[gy, gx] += mw * mh

        # Target density per grid bin
        total_area = sum(
            c["width"] * c["height"] for c in task_data.std_cells
        ) + sum(
            m["width"] * m["height"] for m in task_data.fixed_macros
        )
        avg_density = total_area / (grid_size * grid_size)

        # Compute density gradient via finite differences
        density_grad_x = np.zeros((grid_size, grid_size), dtype=task_data.precision)
        density_grad_y = np.zeros((grid_size, grid_size), dtype=task_data.precision)
        density_grad_x[:, 1:] = density_map[:, 1:] - density_map[:, :-1]
        density_grad_y[1:, :] = density_map[1:, :] - density_map[:-1, :]

        for i in range(n):
            gx, gy = cell_grid[i]
            overflow = density_map[gy, gx] - avg_density
            if overflow > 0:
                force[i, 0] -= density_weight * density_grad_x[gy, gx]
                force[i, 1] -= density_weight * density_grad_y[gy, gx]

        # Adaptive step size: reduce over iterations
        lr = learning_rate * (1.0 - iteration / max_iter)
        pos += lr * force

        # Clamp to canvas bounds
        for i in range(n):
            w_i = task_data.std_cells[i]["width"]
            h_i = task_data.std_cells[i]["height"]
            pos[i, 0] = np.clip(pos[i, 0], w_i / 2, canvas_w - w_i / 2)
            pos[i, 1] = np.clip(pos[i, 1], h_i / 2, canvas_h - h_i / 2)

    task_data.sol = pos
    return task_data


class ForceDirectedSolver(SolverBase):
    """
    Force-Directed Solver for Standard Cell Placement.

    Implements an analytical placement approach combining a spring-based 
    wirelength model with density spreading forces, following the methodology 
    of Eisenmann & Johannes (DAC 1998) and modern ePlace/DREAMPlace frameworks.
    """
    def __init__(
        self,
        max_iter: int = 500,
        learning_rate: float = 0.5,
        density_weight: float = 0.01,
        grid_size: int = 32,
        optimizer: OptimizerBase = None,
    ):
        # Super Initialization
        super(ForceDirectedSolver, self).__init__(
            solver_type=SOLVER_TYPE.FORCE_DIRECTED, optimizer=optimizer
        )

        # Initialize Attributes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.density_weight = density_weight
        self.grid_size = grid_size

    def _solve(self, task_data: TaskBase):
        if task_data.task_type == TASK_TYPE.STANDARD_CELL_PLACEMENT:
            return _force_directed_std_cell_placement(
                task_data=task_data,
                max_iter=self.max_iter,
                learning_rate=self.learning_rate,
                density_weight=self.density_weight,
                grid_size=self.grid_size,
            )
        else:
            raise ValueError(
                f"Solver {self.solver_type} is not supported for {task_data.task_type}."
            )
