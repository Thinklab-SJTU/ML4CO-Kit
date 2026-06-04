r"""
Base class for MILP problems.
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


import pathlib
import numpy as np
import scipy.sparse
import gurobipy as gp
from typing import Union
from ml4co_kit.utils import check_file_path
from ml4co_kit.task.base import TaskBase, TASK_TYPE


class MILPTaskBase(TaskBase):
    def __init__(
        self,
        task_type: TASK_TYPE,
        lp_relaxed: bool,
        minimize: bool,
        precision: Union[np.float32, np.float64] = np.float32,
        threshold: float = 1e-5
    ):
        # Super Initialization
        super(MILPTaskBase, self).__init__(
            task_type=task_type, 
            minimize=minimize, 
            precision=precision
        )

        # Whether the problem is relaxed
        self.lp_relaxed = lp_relaxed

        # Threadhold for floating point precision
        self.threshold = threshold 

        # Initialize Attributes
        self.constrs_num: int = None                  # Number of constraints M
        self.vars_num: int = None                    # Number of variables N
        self.A: scipy.sparse.csr_matrix = None       # CSR format (M, N)
        self.A_coo: scipy.sparse.coo_matrix = None   # COO format (M, N)
        self.A_dense: np.ndarray = None              # Dense format (M, N)
        self.c: np.ndarray = None                    # Objective coefficients (N,)
        self.ls: np.ndarray = None                   # Constraint lower bound (M,)
        self.us: np.ndarray = None                   # Constraint upper bound (M,)
        self.lx: np.ndarray = None                   # Variable lower bound (N,)
        self.ux: np.ndarray = None                   # Variable upper bound (N,)
        self.int_flag: np.ndarray = None               # Whether the variables are integer (N,)
    
    def _check_A_dim(self):
        """
        Checks if the ``A`` attribute is a 2D array.
        """
        if len(self.A.shape) != 2:
            raise ValueError("Constraint matrix ``A`` should be a 2D array.")

    def _check_A_not_none(self):
        """
        Checks if the ``A`` attribute is not None.
        Raises a ``ValueError`` if ``A`` is ``None``. 
        """
        if self.A is None:
            raise ValueError("Constraint matrix ``A`` cannot be None!")

    def _check_c_dim(self):
        """
        Checks if the ``c`` attribute is a 1D array.
        """
        if self.c.ndim != 1:
            raise ValueError("Objective coefficients ``c`` should be a 1D array.")

    def _check_c_not_none(self):
        """
        Checks if the ``c`` attribute is not None.
        Raises a ``ValueError`` if ``c`` is ``None``. 
        """
        if self.c is None:
            raise ValueError("Objective coefficients ``c`` cannot be None!")

    def _check_ls_dim(self):
        """
        Checks if the ``ls`` attribute is a 1D array.
        """
        if self.ls.ndim != 1:
            raise ValueError("Constraint lower bound ``ls`` should be a 1D array.")

    def _check_ls_not_none(self):
        """

        Checks if the ``ls`` attribute is not None.
        Raises a ``ValueError`` if ``ls`` is ``None``. 
        """
        if self.ls is None:
            raise ValueError("Constraint lower bound ``ls`` cannot be None!")

    def _check_us_dim(self):
        """
        Checks if the ``us`` attribute is a 1D array.
        """
        if self.us.ndim != 1:
            raise ValueError("Constraint upper bound ``us`` should be a 1D array.")

    def _check_us_not_none(self):
        """
        Checks if the ``us`` attribute is not None.
        Raises a ``ValueError`` if ``us`` is ``None``. 
        """
        if self.us is None:
            raise ValueError("Constraint upper bound ``us`` cannot be None!")

    def _check_lx_dim(self):
        """
        Checks if the ``lx`` attribute is a 1D array.
        """
        if self.lx.ndim != 1:
            raise ValueError("Variable lower bound ``lx`` should be a 1D array.")

    def _check_lx_not_none(self):
        """
        Checks if the ``lx`` attribute is not None.
        Raises a ``ValueError`` if ``lx`` is ``None``. 
        """
        if self.lx is None:
            raise ValueError("Variable lower bound ``lx`` cannot be None!")

    def _check_ux_dim(self):
        """
        Checks if the ``ux`` attribute is a 1D array.
        """
        if self.ux.ndim != 1:
            raise ValueError("Variable upper bound ``ux`` should be a 1D array.")

    def _check_ux_not_none(self):
        """
        Checks if the ``ux`` attribute is not None.
        Raises a ``ValueError`` if ``ux`` is ``None``. 
        """
        if self.ux is None:
            raise ValueError("Variable upper bound ``ux`` cannot be None!")

    def _check_int_flag_dim(self):
        """
        Checks if the ``int_flag`` attribute is a 1D array.
        """
        if self.int_flag.ndim != 1:
            raise ValueError("``int_flag`` should be a 1D array.")

    def _check_int_flag_not_none(self):
        """
        Checks if the ``int_flag`` attribute is not None.
        Raises a ``ValueError`` if ``int_flag`` is ``None``. 
        """
        if self.int_flag is None:
            raise ValueError("``int_flag`` cannot be None!")

    def _check_sol_dim(self):
        """
        Checks if the ``sol`` attribute is a 1D array.
        """
        if self.sol.ndim != 1:
            raise ValueError("Solution should be a 1D array.")

    def _check_ref_sol_dim(self):
        """
        Checks if the ``ref_sol`` attribute is a 1D array.
        """
        if self.ref_sol.ndim != 1:
            raise ValueError("Reference solution should be a 1D array.")

    def _check_constr_vars_num(self):
        """
        Checks if the number of constraints and variables are consistent.
        """
        # Initialize lists to store the number of constraints and variables
        vars_num_list = list()
        constrs_num_list = list()

        # A: Constraint matrix (M, N), CSR
        if self.A is not None:
            constrs_num, vars_num = self.A.shape
            constrs_num_list.append(constrs_num)
            vars_num_list.append(vars_num)

        # A_coo: Constraint matrix (M, N)
        if self.A_coo is not None:
            constrs_num, vars_num = self.A_coo.shape
            constrs_num_list.append(constrs_num)
            vars_num_list.append(vars_num)

        # A_dense: Constraint matrix (M, N)
        if self.A_dense is not None:
            constrs_num, vars_num = self.A_dense.shape
            constrs_num_list.append(constrs_num)
            vars_num_list.append(vars_num)

        # c: Objective coefficients (N,)
        if self.c is not None:
            vars_num = self.c.shape[0]
            vars_num_list.append(vars_num)

        # ls: Constraint lower bound (M,)
        if self.ls is not None:
            constrs_num = self.ls.shape[0]
            constrs_num_list.append(constrs_num)

        # us: Constraint upper bound (M,)
        if self.us is not None:
            constrs_num = self.us.shape[0]
            constrs_num_list.append(constrs_num)

        # lx: Variable lower bound (N,)
        if self.lx is not None:
            vars_num = self.lx.shape[0]
            vars_num_list.append(vars_num)

        # ux: Variable upper bound (N,)
        if self.ux is not None:
            vars_num = self.ux.shape[0]
            vars_num_list.append(vars_num)

        if not vars_num_list:
            raise ValueError("Cannot infer number of variables.")

        # Check if the number of constraints and variables are consistent
        if constrs_num_list and len(set(constrs_num_list)) > 1:
            raise ValueError("Number of constraints are inconsistent.")
        if len(set(vars_num_list)) > 1:
            raise ValueError("Number of variables are inconsistent.")

        # Set number of constraints and variables
        self.vars_num = vars_num_list[0]
        if constrs_num_list:
            self.constrs_num = constrs_num_list[0]
        elif self.A is not None:
            self.constrs_num = int(self.A.shape[0])
        else:
            self.constrs_num = 0

    def _from_with_gurobi(self, file_path: Union[str, pathlib.Path]):
        # Create Gurobi environment
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()

        # Using Gurobi API to read the MPS file
        gp_model = gp.read(str(file_path), env=env)
        gp_model.update()

        # Get name from Gurobi model
        name = gp_model.ModelName
        
        # Get variables and constraints from Gurobi model
        vars_list = gp_model.getVars()
        constrs = gp_model.getConstrs()
        vars_num = len(vars_list)
        constrs_num = len(constrs)

        # Initialize objective coefficients, variable bounds, and integer variables
        c = np.zeros(vars_num, dtype=self.precision)
        lx = np.zeros(vars_num, dtype=self.precision)
        ux = np.zeros(vars_num, dtype=self.precision)
        int_flag = np.zeros(vars_num, dtype=np.bool_)

        # Get objective coefficients, variable bounds, and integer flags
        for v in vars_list:
            j = v.index
            c[j] = v.Obj
            lx[j] = v.LB if v.LB > -gp.GRB.INFINITY else -np.inf
            ux[j] = v.UB if v.UB < gp.GRB.INFINITY else np.inf
            
            # Set integer variables
            if self.lp_relaxed:
                int_flag[j] = False
            else:
                int_flag[j] = v.VType in (gp.GRB.INTEGER, gp.GRB.BINARY)
        
        # Store optimization direction; store ``c`` as minimization coefficients
        self.minimize = gp_model.ModelSense == gp.GRB.MINIMIZE
        if not self.minimize:
            c = -c

        # Get constraint matrix from Gurobi model
        A = gp_model.getA()

        # Get constraint bounds from Gurobi model
        rhs = np.asarray(gp_model.getAttr("RHS", constrs), dtype=self.precision)
        sense = gp_model.getAttr("Sense", constrs)

        # Get constraint lower and upper bounds
        ls = np.full(constrs_num, -np.inf, dtype=self.precision)
        us = np.full(constrs_num, np.inf, dtype=self.precision)
        for i in range(constrs_num):
            _sense = sense[i]
            _rhs = rhs[i]
            if _sense in (gp.GRB.LESS_EQUAL, "<"):
                us[i] = _rhs
            elif _sense in (gp.GRB.GREATER_EQUAL, ">"):
                ls[i] = _rhs
            elif _sense in (gp.GRB.EQUAL, "="):
                ls[i] = us[i] = _rhs
            else:
                raise ValueError(f"Unknown constraint sense: {_sense!r}")

        # Call ``from_data`` to set attributes
        self.from_data(
            A=A, c=c, ls=ls, us=us, lx=lx, ux=ux, int_flag=int_flag, name=name
        )
        
        # Dispose Gurobi model and environment
        env.dispose()
        gp_model.dispose()

    def _write_with_gurobi(self, file_path: Union[str, pathlib.Path]):
        """Write the instance to MPS or LP via Gurobi."""
        # Check file path
        check_file_path(file_path)

        # Create Gurobi environment
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)
        env.start()

        # Build Gurobi model
        gp_model = self._build_gurobi_model(env)

        # Write Gurobi model to file
        gp_model.write(str(file_path))

        # Dispose Gurobi model and environment
        gp_model.dispose()
        env.dispose()

    def _build_gurobi_model(self, env: gp.Env) -> gp.Model:
        """Build a Gurobi model from internal MILP data."""
        # Check Attributes
        self._check_A_not_none()
        self._check_c_not_none()
        self._check_ls_not_none()
        self._check_us_not_none()
        self._check_lx_not_none()
        self._check_ux_not_none()
        self._check_int_flag_not_none()

        # Create Gurobi model
        gp_model = gp.Model(self.name, env=env)
        gp_model.ModelSense = (
            gp.GRB.MINIMIZE if self.minimize else gp.GRB.MAXIMIZE
        )
        obj_coef = self.c if self.minimize else -self.c

        # Set variable types
        vtypes = np.full(self.vars_num, gp.GRB.CONTINUOUS, dtype=object)
        if not self.lp_relaxed:
            binary_mask = (
                self.int_flag
                & (self.lx >= -self.threshold)
                & (self.ux <= 1.0 + self.threshold)
            )
            integer_mask = self.int_flag & (~binary_mask)
            vtypes[binary_mask] = gp.GRB.BINARY
            vtypes[integer_mask] = gp.GRB.INTEGER

        # Set variable bounds
        lx = self.lx.astype(np.float64, copy=False)
        ux = self.ux.astype(np.float64, copy=False)
        lb = np.where(np.isfinite(lx), lx, -gp.GRB.INFINITY)
        ub = np.where(np.isfinite(ux), ux, gp.GRB.INFINITY)
        x = gp_model.addMVar(
            shape=self.vars_num, lb=lb, ub=ub, vtype=vtypes, name="x",
        )

        # Set objective function
        gp_model.setObjective(obj_coef @ x)

        # Add constraints
        A_blocks = []
        rhs_blocks = []
        sense_blocks = []
        eq_mask = (
            np.isfinite(self.ls)
            & np.isfinite(self.us)
            & (np.abs(self.ls - self.us) <= self.threshold)
        )
        le_mask = np.isfinite(self.us) & (~eq_mask)
        ge_mask = np.isfinite(self.ls) & (~eq_mask)

        # Add equality constraints
        if np.any(eq_mask):
            A_blocks.append(self.A[eq_mask])
            rhs_blocks.append(self.us[eq_mask])
            sense_blocks.extend([gp.GRB.EQUAL] * int(np.sum(eq_mask)))

        # Add less than or equal to constraints
        if np.any(le_mask):
            A_blocks.append(self.A[le_mask])
            rhs_blocks.append(self.us[le_mask])
            sense_blocks.extend([gp.GRB.LESS_EQUAL] * int(np.sum(le_mask)))

        # Add greater than or equal to constraints
        if np.any(ge_mask):
            A_blocks.append(self.A[ge_mask])
            rhs_blocks.append(self.ls[ge_mask])
            sense_blocks.extend([gp.GRB.GREATER_EQUAL] * int(np.sum(ge_mask)))

        # Add constraints to Gurobi model
        if len(A_blocks) > 0:
            A_all = scipy.sparse.vstack(A_blocks, format="csr")
            rhs_all = np.concatenate(rhs_blocks)
            sense_all = np.asarray(sense_blocks, dtype="U1")
            gp_model.addMConstr(A=A_all, x=x, sense=sense_all, b=rhs_all)
        gp_model.update()

        # Return Gurobi model
        return gp_model

    def from_data(
        self,
        A: scipy.sparse.csr_matrix = None,
        c: np.ndarray = None,
        ls: np.ndarray = None,
        us: np.ndarray = None,
        lx: np.ndarray = None,
        ux: np.ndarray = None,
        int_flag: np.ndarray = None,
        sol: np.ndarray = None,
        ref: bool = False,
        name: str = None,
    ):
        # Set Attributes and Check Dimensions
        if A is not None:
            self.A_coo = None
            self.A_dense = None
            if scipy.sparse.issparse(A):
                self.A = A.tocsr().astype(self.precision)
            else:
                raise TypeError(
                    "``A`` must be a ``scipy.sparse.spmatrix``; "
                    "use ``from_A_dense`` for dense arrays."
                )
        if c is not None:
            self.c = c.astype(self.precision)
            self._check_c_dim()
        if ls is not None:
            self.ls = ls.astype(self.precision)
            self._check_ls_dim()
        if us is not None:
            self.us = us.astype(self.precision)
            self._check_us_dim()
        if lx is not None:
            self.lx = lx.astype(self.precision)
            self._check_lx_dim()
        if ux is not None:
            self.ux = ux.astype(self.precision)
            self._check_ux_dim()
        if int_flag is not None:
            self.int_flag = int_flag.astype(np.bool_)
            self._check_int_flag_dim()
        if sol is not None:
            if ref:
                self.ref_sol = sol.astype(self.precision)
                self._check_ref_sol_dim()
            else:
                self.sol = sol.astype(self.precision)
                self._check_sol_dim()

        # Check numbers of constraints and variables
        self._check_constr_vars_num()

        # Set Name if Provided
        if name is not None:
            self.name = name
    
    def from_A_coo(self, A_coo: scipy.sparse.coo_matrix):
        # Transfer to CSR format
        A_csr = A_coo.tocsr()

        # Call ``from_data`` to set attributes
        self.from_data(A=A_csr)

        # Set ``A_coo`` attribute
        self.A_coo = A_coo

    def to_A_coo(self) -> scipy.sparse.csr_matrix:
        # Check if ``A_coo`` attribute is None
        if self.A_coo is None:
            # Transfer to COO format
            self.A_coo = self.A.tocoo()

        # Return ``A_coo`` attribute
        return self.A_coo

    def from_A_dense(self, A_dense: np.ndarray):
        # Convert to CSR format
        A_csr = scipy.sparse.csr_matrix(A_dense, dtype=self.precision)

        # Call ``from_data`` to set attributes
        self.from_data(A=A_csr)

        # Set ``A_dense`` attribute
        self.A_dense = A_dense

    def to_A_dense(self) -> np.ndarray:
        # Check if ``A_dense`` attribute is None
        if self.A_dense is None:
            # Transfer to Dense format
            self.A_dense = self.A.toarray()

        # Return ``A_dense`` attribute
        return self.A_dense

    def from_mps(self, file_path: Union[str, pathlib.Path]):
        self._from_with_gurobi(file_path)

    def to_mps(self, file_path: Union[str, pathlib.Path]):
        """Write the instance to an MPS file via Gurobi."""
        self._write_with_gurobi(file_path)

    def from_lp(self, file_path: Union[str, pathlib.Path]):
        self._from_with_gurobi(file_path)

    def to_lp(self, file_path: Union[str, pathlib.Path]):
        """Write the instance to an LP file via Gurobi."""
        self._write_with_gurobi(file_path)

    def check_constraints(self, sol: np.ndarray) -> bool:
        """
        Checks if the solution is valid.
        """
        # Check Constraints: ls <= Ax <= us
        Ax = self.A @ sol
        if not np.all(Ax >= self.ls - self.threshold):
            return False
        if not np.all(Ax <= self.us + self.threshold):
            return False

        # Check Variables: lx <= x <= ux
        if not np.all(sol >= self.lx - self.threshold):
            return False
        if not np.all(sol <= self.ux + self.threshold):
            return False
        
        # Check Integer Variables: int_flag
        int_sol = sol[self.int_flag]
        int_sol_diff = np.abs(int_sol - np.round(int_sol))
        if not np.all(int_sol_diff <= self.threshold):
            return False
        
        # If all constraints and variables are satisfied, return True
        return True

    def evaluate(self, sol: np.ndarray, check_constr: bool = True) -> np.floating:
        """
        Evaluates the objective function.
        """
        # Check Constraints
        if check_constr and not self.check_constraints(sol):
            raise ValueError("Invalid solution!")
        
        # Evaluate the objective function
        obj = np.dot(self.c, sol)
        return obj