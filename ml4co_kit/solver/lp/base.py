r"""
Basic solver for Linear Program (LP). 

LP is a mathematical optimization technique used to find the best outcome in a given mathematical model. 
It involves maximizing or minimizing a linear objective function subject to linear constraints.
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
from ml4co_kit.solver.base import SolverBase
from ml4co_kit.utils.time_utils import iterative_execution_for_file
from ml4co_kit.utils.type_utils import to_numpy, TASK_TYPE, SOLVER_TYPE


class LPSolver(SolverBase):
    r"""
    min cx, st wx=b 
    """
    def __init__(self, solver_type: SOLVER_TYPE = None, time_limit: float = 60.0):
        super(LPSolver, self).__init__(
            task_type=TASK_TYPE.LP, solver_type=solver_type
        )
        self.time_limit = time_limit
        self.w: np.ndarray = None
        self.c: np.ndarray = None
        self.b: np.ndarray = None
        self.x: np.ndarray = None
        self.ref_x: np.ndarray = None

    def _check_w_dim(self):
        r"""
        Ensures that the ``w`` attribute is a 3D array. If ``w`` is a 2D array,
        it adds an additional dimension to make it 3D. Raises a ``ValueError`` if ``points``
        is neither 2D nor 3D.
        """
        if self.w is not None:
            if self.w.ndim == 2:
                self.w = np.expand_dims(self.w, axis=0)
            if self.w.ndim != 3:
                raise ValueError("``w`` must be a 2D or 3D array.")

    def _check_c_dim(self):
        r"""
        Ensures that the ``c`` attribute is a 2D array. If ``c`` is a 1D array,
        it adds an additional dimension to make it 2D. Raises a ``ValueError`` if ``c``
        has more than 2 dimensions.
        """
        if self.c is not None:
            if self.c.ndim == 1:
                self.c = np.expand_dims(self.c, axis=0)
            if self.c.ndim != 2:
                raise ValueError("The dimensions of ``c`` cannot be larger than 2.")

    def _check_b_dim(self):
        r"""
        Ensures that the ``b`` attribute is a 2D array. If ``b`` is a 1D array,
        it adds an additional dimension to make it 2D. Raises a ``ValueError`` if ``b``
        has more than 2 dimensions.
        """
        if self.b is not None:
            if self.b.ndim == 1:
                self.b = np.expand_dims(self.b, axis=0)
            if self.b.ndim != 2:
                raise ValueError("The dimensions of ``b`` cannot be larger than 2.")

    def _check_x_dim(self):
        r"""
        Ensures that the ``x`` attribute is a 2D array. If ``x`` is a 1D array,
        it adds an additional dimension to make it 2D. Raises a ``ValueError`` if ``x``
        has more than 2 dimensions.
        """
        if self.x is not None:
            if self.x.ndim == 1:
                self.x = np.expand_dims(self.x, axis=0)
            if self.x.ndim != 2:
                raise ValueError("The dimensions of ``x`` cannot be larger than 2.")

    def _check_ref_x_dim(self):
        r"""
        Ensures that the ``ref_x`` attribute is a 2D array. If ``ref_x`` is a 1D array,
        it adds an additional dimension to make it 2D. Raises a ``ValueError`` if ``ref_x``
        has more than 2 dimensions.
        """
        if self.ref_x is not None:
            if self.ref_x.ndim == 1:
                self.ref_x = np.expand_dims(self.ref_x, axis=0)
            if self.ref_x.ndim != 2:
                raise ValueError("The dimensions of ``ref_x`` cannot be larger than 2.")

    def _check_w_not_none(self):
        r"""
        Checks if the ``w`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``w`` is ``None``. 
        """
        if self.w is None:
            message = (
                "``w`` cannot be None! You can load the ``w`` using the methods"
                "``from_data`` or ``from_txt``."
            )
            raise ValueError(message)

    def _check_c_not_none(self):
        r"""
        Checks if the ``c`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``c`` is ``None``. 
        """
        if self.c is None:
            message = (
                "``c`` cannot be None! You can load the ``c`` using the methods"
                "``from_data`` or ``from_txt``."
            )
            raise ValueError(message)
        
    def _check_b_not_none(self):
        r"""
        Checks if the ``b`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``b`` is ``None``. 
        """
        if self.c is None:
            message = (
                "``b`` cannot be None! You can load the ``b`` using the methods"
                "``from_data`` or ``from_txt``."
            )
            raise ValueError(message)

    def _check_x_not_none(self, ref: bool):
        r"""
        Checks if the ``x` or ``ref_x`` attribute is not ``None``.
        - If ``ref`` is ``True``, it checks the ``ref_x`` attribute.
        - If ``ref`` is ``False``, it checks the ``x`` attribute.
        Raises a `ValueError` if the respective attribute is ``None``.
        """
        msg = "ref_x" if ref else "x"
        message = (
            f"``{msg}`` cannot be None! You can use solvers based on "
            "``LPSolver`` like ``LPGurobiSolver`` or use methods including "
            "``from_data`` or ``from_txt`` to obtain them."
        )
        if ref:
            if self.ref_x is None:
                raise ValueError(message)
        else:
            if self.x is None:    
                raise ValueError(message)

    def _check_constraints(self, ref: bool):
        r"""
        Checks whether constraints are met.
        - If ``ref`` is ``True``, it checks the ``ref_x`` attribute.
        - If ``ref`` is ``False``, it checks the ``x`` attribute.
        Raises a `ValueError` if the corresponding constraint is not satisfied.
        """
        self._check_w_not_none()
        self._check_b_not_none()
        msg = "ref_x" if ref else "x"
        message = (
            f"``{msg}`` does not meet the constraint. Please carefully check whether "
            "there is a problem in the generation of Solutions."
        )
        if ref:
            wx = np.squeeze(np.matmul(self.w, np.expand_dims(self.ref_x, -1)), -1)  
        else:
            wx = np.squeeze(np.matmul(self.w, np.expand_dims(self.x, -1)), -1)
        if np.max(np.abs(wx - self.b)) > 1e-3:
            raise ValueError(message)

    def from_txt(
        self,
        file_path: str,
        ref: bool = False,
        return_list: bool = False,
        show_time: bool = False
    ):
        # check the file format
        if not file_path.endswith(".txt"):
            raise ValueError("Invalid file format. Expected a ``.txt`` file.")

        # read the data form .txt
        with open(file_path, "r") as file:
            w_list = list()
            c_list = list()
            b_list = list()
            x_list = list()
            load_msg = f"Loading data from {file_path}"
            for line in iterative_execution_for_file(file, load_msg, show_time):
                # get data
                line = line.strip()
                split_lines_1 = line.split("  c ")
                str_w = split_lines_1[0][2:]
                split_lines_2 = split_lines_1[1].split(" b ")
                str_c = split_lines_2[0]
                split_lines_3 = split_lines_2[1].split(" output ")
                str_b = split_lines_3[0]
                str_x = split_lines_3[1]
                
                # np.ndarray
                str_w = str_w.split(" ")
                str_w.append('')
                w = np.array([float(str_w[2*i]) for i in range(len(str_w) // 2)])
                str_c = str_c.split(" ")
                c = np.array(str_c).astype(np.float32)
                str_b = str_b.split(" ")
                b = np.array(str_b).astype(np.float32)
                str_x = str_x.split(" ")
                x = np.array(str_x).astype(np.float32)
                constr_num = len(b)
                vars_num = len(c)
                w = w.reshape(constr_num, vars_num)
                
                # add to list
                w_list.append(w)
                c_list.append(c)
                b_list.append(b)
                x_list.append(x)

        if return_list:
            return w_list, c_list, b_list, x_list

        try:
            w = np.array(w_list)
            c = np.array(c_list)
            b = np.array(b_list)
            x = np.array(x_list)
        except Exception as e:
            message = (
                "This method does not support instances of different numbers of variables and constraints. "
                "If you want to read the data, please set ``return_list`` as True. "
                "Anyway, the data will not be saved in the solver. "
                "Please convert the data to ``np.ndarray`` externally before calling the solver."
            )
            raise Exception(message) from e

        self.from_data(w=w, c=c, b=b, x=x, ref=ref)
        
    def from_data(
        self,
        w: Union[list, np.ndarray] = None,
        c: Union[list, np.ndarray] = None,
        b: Union[list, np.ndarray] = None,
        x: Union[list, np.ndarray] = None,
        ref: bool = False
    ):
        # read data
        if w is not None:
            self.w = to_numpy(w).astype(np.float32)
            self._check_w_dim()
            
        if c is not None:
            self.c = to_numpy(c).astype(np.float32)
            self._check_c_dim()
            
        if b is not None:
            self.b = to_numpy(b).astype(np.float32)
            self._check_b_dim()
            
        if x is not None:
            x = to_numpy(x).astype(np.float32)
            if ref:
                self.ref_x = x
                self._check_ref_x_dim()
            else:
                self.x = x
                self._check_x_dim()             

    def to_txt(self, file_path: str = "example.txt"):
        # check
        self._check_w_not_none()
        self._check_c_not_none()
        self._check_b_not_none()
        self._check_x_not_none(ref=False)

        # write
        for _w, _c, _b, _sol in zip(self.w, self.c, self.b, self.x):
            with open(file_path, "a+") as f:
                f.write(str("w") + str(" "))
                for line in _w:
                    f.write(" ".join(str(x) + str(" ") for x in line))
                    f.write(str(" "))
                f.write("c " + str(" ").join(str(cc) for cc in _c))
                f.write(" b " + str(" ").join(str(bb) for bb in _b))
                f.write(str(" output") + str(" "))
                f.write(str(" ").join(str(xx) for xx in _sol))
                f.write("\n")
            f.close()

    def evaluate(self, check_constraints: bool = True, calculate_gap: bool = False):
        # check
        if check_constraints:
            self._check_constraints(ref=False)
        self._check_c_not_none()
        self._check_x_not_none(ref=False)
        if calculate_gap:
            self._check_x_not_none(ref=True)
            if check_constraints:
                self._check_constraints(ref=True)

        # prepare for evaluate
        x_cost_list = list()
        samples = self.w.shape[0]
        if calculate_gap:
            ref_x_cost_list = list()
            gap_list = list()

        # deal with different situation
        for idx in range(samples):
            solved_cost = np.sum(np.multiply(self.x[idx], self.c[idx]))
            x_cost_list.append(solved_cost)
            if calculate_gap:
                ref_cost = np.sum(np.multiply(self.ref_x[idx], self.c[idx]))
                ref_x_cost_list.append(ref_cost)
                gap = (solved_cost - ref_cost) / ref_cost * 100
                gap_list.append(gap)

        # calculate average cost/gap & std
        x_costs = np.array(x_cost_list)
        if calculate_gap:
            ref_costs = np.array(ref_x_cost_list)
            gaps = np.array(gap_list)
        costs_avg = np.average(x_costs)
        if calculate_gap:
            ref_costs_avg = np.average(ref_costs)
            gap_avg = np.sum(gaps) / samples
            gap_std = np.std(gaps)
            return costs_avg.item(), ref_costs_avg.item(), gap_avg.item(), gap_std.item()
        else:
            return costs_avg.item()
 
    def solve(
        self,
        w: Union[list, np.ndarray] = None,
        c: Union[list, np.ndarray] = None,
        b: Union[list, np.ndarray] = None,
        num_threads: int = 1,
        show_time: bool = False,
        **kwargs,
    ) -> np.ndarray:
        raise NotImplementedError(
            "The ``solve`` function is required to implemented in subclasses."
        )

    def __str__(self) -> str:
        return "LPSolver"