r"""
Graph Matching Tools.
"""

# Copyright (c) 2022 Thinklab@SJTU
# pygmtools is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
# http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.


from torch import Tensor
from .astar import pygm_astar
from .hungarian import pygm_hungarian
from .ipfp import pygm_ipfp
from .rrwm import pygm_rrwm
from .sinkhorn import pygm_sinkhorn
from .sm import pygm_sm


class PyGMToolsQAPSolver(object):
    def __init__(
        self,
        solver_name = "rrwm", 
        astar_beam_width: int = 0,
        is_dummy: bool = False,
        ipfp_x0: Tensor = None,
        ipfp_max_iter: int = 50,
        rrwm_x0: Tensor = None,
        rrwm_max_iter: int = 50,
        rrwm_sk_iter: int = 10,
        rrwm_alpha: float = 0.2,
        rrwm_beta: float = 30,
        sm_x0: Tensor = None,
        sm_max_iter: int = 50
    ):
        # Initialize Attributes (Solver Name)
        self.solver_name = solver_name

        # Initialize Attributes (ASTAR)
        self.astar_beam_width = astar_beam_width
        self.is_dummy = is_dummy

        # Initialize Attributes (IPFP)
        self.ipfp_x0 = ipfp_x0
        self.ipfp_max_iter = ipfp_max_iter

        # Initialize Attributes (RRWM)
        self.rrwm_x0 = rrwm_x0
        self.rrwm_max_iter = rrwm_max_iter
        self.rrwm_sk_iter = rrwm_sk_iter
        self.rrwm_alpha = rrwm_alpha
        self.rrwm_beta = rrwm_beta

        # Initialize Attributes (SM)
        self.sm_x0 = sm_x0
        self.sm_max_iter = sm_max_iter
    
    def solve(self, K: Tensor, n1: Tensor, n2: Tensor, n1max: int, n2max: int):
        if self.solver_name == "astar":
            return pygm_astar(
                K=K, n1=n1, n2=n2, n1max=n1max, n2max=n2max, 
                beam_width=self.astar_beam_width,
                is_dummy=self.is_dummy
            )
        elif self.solver_name == "ipfp":
            return pygm_ipfp(
                K=K, n1=n1, n2=n2, n1max=n1max, n2max=n2max, 
                x0=self.ipfp_x0, max_iter=self.ipfp_max_iter
            )
        elif self.solver_name == "rrwm":
            return pygm_rrwm(
                K=K, n1=n1, n2=n2, n1max=n1max, n2max=n2max, 
                x0=self.rrwm_x0, max_iter=self.rrwm_max_iter, 
                sk_iter=self.rrwm_sk_iter, alpha=self.rrwm_alpha, 
                beta=self.rrwm_beta
            )
        elif self.solver_name == "sm":
            return pygm_sm(
                K=K, n1=n1, n2=n2, n1max=n1max, n2max=n2max, 
                x0=self.sm_x0, max_iter=self.sm_max_iter
            )
        else:
            raise ValueError(f"Solver {self.solver_name} is not supported!")