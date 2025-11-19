r"""
PySAT Solver for SAT-P
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


from pysat.solvers import Glucose3
from ml4co_kit.task.sat.satp import SATPTask


def satp_pysat(task_data: SATPTask):
    solver = Glucose3()
    
    for clause in task_data.clauses:
        solver.add_clause(clause)
    
    satisfiable = solver.solve()
    solver.delete()
    
    task_data.sol = satisfiable
