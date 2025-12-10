r"""
PySAT Solver for UNSAT-C
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
from pysat.formula import CNF
from pysat.examples.musx import MUSX
from ml4co_kit.task.sat.unsatc import USATCTask


def unsatc_pysat(task_data: USATCTask, solver_name: str):
    # Create CNF object
    cnf = CNF()
    cnf.clauses = task_data.clauses
    cnf.nv = task_data.vars_num
    
    # Using MUSX to compute the unsat core
    musx = MUSX(cnf, solver=solver_name, verbosity=0)
    core = musx.compute()
    
    # Get the unsat core variables
    sol = np.zeros(task_data.vars_num, dtype=bool)
    if core:
        for lit in core:
            var_idx = abs(lit) - 1
            if var_idx < task_data.vars_num:
                sol[var_idx] = True
    
    # Delete the MUSX object
    musx.delete()
    
    # Store the solution
    task_data.from_data(sol=sol, ref=False)
