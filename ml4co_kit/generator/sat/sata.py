r"""
Generator for SAT-A (Satisfying Assignment Prediction) task instances.
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


import numpy as np
import pysat.solvers
from typing import Union, Callable
from ml4co_kit.task.sat import SATATask
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.generator.sat.base import SATGeneratorBase, SAT_TYPE


class SATAGenerator(SATGeneratorBase):
    """Generator for SAT-A (Satisfying Assignment Prediction) task instances."""
    
    def __init__(
        self,
        distribution_type: SAT_TYPE = SAT_TYPE.PHASE,
        precision: Union[np.float32, np.float64] = np.float32,
        # special args for phase
        phase_n_range: tuple = (10, 40),
        phase_k: int = 3,
        phase_alpha: float = 4.26,
        # special args for sr
        sr_n_range: tuple = (10, 40),
        sr_b: float = 0.3,
        sr_g: float = 0.4,
        # special args for ca
        ca_n_range: tuple = (10, 40),
        ca_mn_range: tuple = (13, 15),
        ca_k_range: tuple = (4, 5),
        ca_c_range: tuple = (3, 10),
        ca_q_range: tuple = (0.7, 0.9),
        # special args for ps
        ps_n_range: tuple = (10, 40),
        ps_mn_range: tuple = (6, 8),
        ps_k_range: tuple = (4, 5),
        ps_beta_range: tuple = (0.0, 1.0),
        ps_beta_prime: float = 1.0,
        ps_t_range: tuple = (0.75, 1.5),
        # special args for k_clique
        k_clique_v_range: tuple = (15, 20),
        k_clique_k_range: tuple = (3, 5),
        # special args for k_domset
        k_domset_v_range: tuple = (15, 20),
        k_domset_k_range: tuple = (3, 5),
        # special args for k_vercov
        k_vercov_v_range: tuple = (10, 20),
        k_vercov_k_range: tuple = (6, 8),
        # base solver
        base_solver: str = "cadical195"
    ):
        # Super Initialization
        super(SATAGenerator, self).__init__(
            task_type=TASK_TYPE.SATA,
            distribution_type=distribution_type,
            precision=precision,
            phase_n_range=phase_n_range,
            phase_k=phase_k,
            phase_alpha=phase_alpha,
            sr_n_range=sr_n_range,
            sr_b=sr_b,
            sr_g=sr_g,
            ca_n_range=ca_n_range,
            ca_mn_range=ca_mn_range,
            ca_k_range=ca_k_range,
            ca_c_range=ca_c_range,
            ca_q_range=ca_q_range,
            ps_n_range=ps_n_range,
            ps_k_range=ps_k_range,
            ps_mn_range=ps_mn_range,
            ps_beta_range=ps_beta_range,
            ps_beta_prime=ps_beta_prime,
            ps_t_range=ps_t_range,
            k_clique_v_range=k_clique_v_range,
            k_clique_k_range=k_clique_k_range,
            k_domset_v_range=k_domset_v_range,
            k_domset_k_range=k_domset_k_range,
            k_vercov_v_range=k_vercov_v_range,
            k_vercov_k_range=k_vercov_k_range,
            base_solver=base_solver
        )

    def _idx2bool(self, idx: np.ndarray) -> np.ndarray:
        bool_sol = np.zeros_like(idx, dtype=np.bool_)
        bool_sol[idx > 0] = True
        bool_sol[idx < 0] = False
        return bool_sol
    
    def _create_instance(self, gen_func: Callable) -> SATATask:
        # Generate SAT clauses (until the clauses are satisfiable)
        while True:
            clauses = gen_func()
            solver = pysat.solvers.Solver(self.base_solver, bootstrap_with=clauses)
            if solver.solve():
                sol = solver.get_model()
                break
        
        # Create SAT-A task
        task_data = SATATask(precision=self.precision)
        bool_sol = self._idx2bool(np.array(sol))
        task_data.from_data(clauses=clauses, sol=bool_sol, ref=True)
        return task_data

    def _generate_phase(self) -> SATATask:
        return self._create_instance(self._super_generate_phase)

    def _generate_sr(self) -> SATATask:
        # Call `_super_generate_sr` to generate SAT clauses
        _, sat_clauses, sat_sol = self._super_generate_sr()
        
        # Create SAT-A task
        task_data = SATATask(precision=self.precision)
        bool_sol = self._idx2bool(np.array(sat_sol))
        task_data.from_data(clauses=sat_clauses, sol=bool_sol, ref=True)
        return task_data

    def _generate_ca(self) -> SATATask:
        return self._create_instance(self._super_generate_ca)

    def _generate_ps(self) -> SATATask:
        return self._create_instance(self._super_generate_ps)

    def _generate_k_clique(self) -> SATATask:
        return self._create_instance(self._super_generate_k_clique)

    def _generate_k_domset(self) -> SATATask:
        return self._create_instance(self._super_generate_k_domset)

    def _generate_k_vercov(self) -> SATATask:
        return self._create_instance(self._super_generate_k_vercov)