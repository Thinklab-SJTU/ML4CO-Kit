r"""
PySAT Solver for SAT-A
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
from pysat.solvers import Glucose3
from ml4co_kit.task.sat.sata import SATATask


def sata_pysat(task_data: SATATask):
    solver = Glucose3()
    
    for clause in task_data.clauses:
        solver.add_clause(clause)
    
    if solver.solve():
        model = solver.get_model()
        sol = np.zeros(task_data.vars_num, dtype=bool)
        for lit in model:
            var_idx = abs(lit) - 1
            if var_idx < task_data.vars_num:
                sol[var_idx] = (lit > 0)
    else:
        sol = np.zeros(task_data.vars_num, dtype=bool)
    
    solver.delete()
    task_data.sol = sol
