r"""
Concorde Algorithm for TSP
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
import numpy as np
from ml4co_kit.task.routing.tsp import TSPTask
from ml4co_kit.solver.lib.concorde.pyconcorde import TSPConSolver


def tsp_concorde(task_data: TSPTask, concorde_scale: int=1e6):

    
    # Get the data
    points = task_data.points * concorde_scale
    name = task_data.name
    norm = task_data.distance_type.value
    
    # Create the solver
    solver = TSPConSolver.from_data(
        xs=points[:, 0], ys=points[:, 1], norm=norm, name=name,
    )
    solution = solver.solve(verbose=False, name=name)
    tour: np.ndarray = solution.tour
    tour = np.append(tour, tour[0])
    
    # Store the tour in the task_data
    task_data.from_data(sol=tour, ref=False)
    
    # Prepare for cleanup
    _name = name[0:9]
    sol_filename = f"{_name}.sol"
    Osol_filename = f"O{_name}.sol"
    res_filename = f"{_name}.res"
    Ores_filename = f"O{_name}.res"
    sav_filename = f"{_name}.sav"
    Osav_filename = f"O{_name}.sav"
    pul_filename = f"{_name}.pul"
    Opul_filename = f"O{_name}.pul"
    filelist = [
        sol_filename,
        Osol_filename,
        res_filename,
        Ores_filename,
        sav_filename,
        Osav_filename,
        pul_filename,
        Opul_filename,
    ]
    for i in range(100):
        filelist.append("{}.{:03d}".format(name[0:8], i + 1))
    
    # Clean up
    for file in filelist:
        if os.path.exists(file):
            os.remove(file)
