r"""
Koopmans-Beckmann Quadratic Assignment Problem (KQAP).

The Koopmans-Beckmann QAP is a classic formulation where we need to assign n facilities 
to n locations. The objective is to minimize the total cost, which is the sum of:
1. Flow between facilities i and j multiplied by distance between locations p and q
2. Cost of assigning facility i to location p

Mathematically: min sum_{i,j} F[i,j] * D[p,q] + sum_i C[i,p]
where the assignment is represented by a permutation matrix X.
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
from typing import Union
from ml4co_kit.task.base import TASK_TYPE
from ml4co_kit.task.qap.base import QAPTaskBase


class KQAPTask(QAPTaskBase):
    def __init__(
        self,
        precision: Union[np.float32, np.float64] = np.float32
    ):
        super(KQAPTask, self).__init__(
            task_type=TASK_TYPE.KQAP, 
            minimize=True, 
            precision=precision
        )

        self.F = None  # flow matrix (n1, n1)
        self.D = None  # distance matrix (n2, n2)

    def from_data(
        self, 
        F: np.ndarray = None, 
        D: np.ndarray = None,
        sol: np.ndarray = None,
        ref: bool = False,
    ) -> None:
        # Set Attributes
        if F is not None:
            self.F = F.astype(self.precision)
            n1: int = F.shape[0]
        if D is not None:
            self.D = D.astype(self.precision)
            n2: int = D.shape[0]

        # Build Affinity Matrix
        if F is not None or D is not None:
            K = np.kron(self.F, self.D)
        else:
            K = None
        
        # Call super ``from_data``
        super().from_data(K=K, n1=n1, n2=n2, sol=sol, ref=ref)