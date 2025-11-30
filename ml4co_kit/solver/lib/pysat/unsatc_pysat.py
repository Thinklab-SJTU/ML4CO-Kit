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


def unsatc_pysat(task_data: USATCTask):
    cnf = CNF()
    cnf.clauses = task_data.clauses
    cnf.nv = task_data.vars_num
    
    musx = MUSX(cnf, verbosity=0)
    core = musx.compute()
    
    sol = np.zeros(task_data.vars_num, dtype=bool)
    if core:
        for lit in core:
            var_idx = abs(lit) - 1
            if var_idx < task_data.vars_num:
                sol[var_idx] = True
    
    musx.delete()
    task_data.sol = sol
