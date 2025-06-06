r"""
Basic solver for Knapsack Problem (KP). 

In the KP problem, you need to pack a set of items, with given values and weights, into a container with a maximum capacity. 
The problem is to choose a subset of the items of maximum total value that will fit in the container.
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
from ml4co_kit.utils.type_utils import to_numpy, TASK_TYPE, SOLVER_TYPE
from ml4co_kit.utils.time_utils import iterative_execution_for_file


class KPSolver(SolverBase):
    r"""
    This class provides a basic framework for solving knapsack problems. It includes methods for 
    loading and outputting data in various file formats and evaluating solutions. 
    Note that the actual solving method should be implemented in subclasses.

    :param items_num: :math:`N`, int, the number of items in KP problem.
    :param scale: int, the scale of the weights and values, by which the final objective should be divided.
    :param weights: np.ndarray, the weight of each item for selection.
    :param capacities: np.ndarray, the capacity of the knapsacks.
    :param values: np.ndarray, the value of each item for selection.
    :param items_label: np.ndarray, the solutions to the problems (a binary vector indicating the selection of items).
    :param ref_items_label: np.ndarray, the reference solutions to the problems.
    :params sel_items_value: np.ndarray, the objective (total values of the selected items) to the problms.
    :params ref_sel_items_value: np.ndarray, the reference objective to the problms.
    """
    def __init__(
        self, 
        solver_type: SOLVER_TYPE = None, 
        scale: int = 1e6,
        time_limit: float = 60.0
    ):
        super(KPSolver, self).__init__(
            task_type=TASK_TYPE.KP, solver_type=solver_type
        )
        self.solver_type = solver_type
        self.scale = scale
        self.time_limit = time_limit
        self.weights: np.ndarray = None
        self.values: np.ndarray = None
        self.capacities: np.ndarray = None
        self.items_label: np.ndarray = None
        self.ref_items_label: np.ndarray = None
        self.sel_items_value: np.ndarray = None
        self.ref_sel_items_value: np.ndarray = None
        
    def _check_weights_dim(self):
        r"""
        Ensures that the ``weights`` attribute is a 2D array. If ``weights`` is a 1D array,
        it adds an additional dimension to make it 2D. Raises a ``ValueError`` if ``weights``
        is neither 1D nor 2D. 
        """
        if self.weights is not None:
            if self.weights.ndim == 1:
                self.weights = np.expand_dims(self.weights, axis=0)
            if self.weights.ndim != 2:
                raise ValueError("The dimensions of ``weights`` cannot be larger than 2.")
            
    def _check_values_dim(self):
        r"""
        Ensures that the ``values`` attribute is a 2D array. If ``values`` is a 1D array,
        it adds an additional dimension to make it 2D. Raises a ``ValueError`` if ``values``
        is neither 1D nor 2D. 
        """
        if self.values is not None:
            if self.values.ndim == 1:
                self.values = np.expand_dims(self.values, axis=0)
            if self.values.ndim != 2:
                raise ValueError("The dimensions of ``values`` cannot be larger than 2.")
    
    def _check_capacities_dim(self):
        r"""
        Ensures that the ``capacities`` attribute is a 1D array. Raises a ``ValueError`` 
        if ``capacities`` is not 1D.
        """
        if self.capacities is not None:
            if self.capacities.ndim != 1:
                raise ValueError("The ``capacities`` must be 1D array.")
    
    def _check_items_label_dim(self):
        r"""
        Ensures that the ``items_label`` attribute is a 2D array. If ``items_label`` is a 1D array,
        it adds an additional dimension to make it 2D. Raises a ``ValueError`` if ``items_label``
        has more than 2 dimensions.
        """
        if self.items_label is not None:
            if self.items_label.ndim == 1:
                self.items_label = np.expand_dims(self.items_label, axis=0)
            if self.items_label.ndim != 2:
                raise ValueError("The dimensions of ``items_label`` cannot be larger than 2.")

    def _check_ref_items_label_dim(self):
        r"""
        Ensures that the ``ref_items_label`` attribute is a 2D array. If ``ref_items_label`` is a 1D array,
        it adds an additional dimension to make it 2D. Raises a ``ValueError`` if ``ref_items_label``
        has more than 2 dimensions.
        """
        if self.ref_items_label is not None:
            if self.ref_items_label.ndim == 1:
                self.ref_items_label = np.expand_dims(self.ref_items_label, axis=0)
            if self.ref_items_label.ndim != 2:
                raise ValueError(
                    "The dimensions of the ``ref_items_label`` cannot be larger than 2."
                )

    def _check_weights_not_none(self):
        r"""
        Checks if the ``weights`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``weights`` is ``None``. 
        """
        if self.weights is None:
            message = (
                "``weights`` cannot be None! You can load the instances using the methods"
                "``from_data`` or ``from_txt``."
            )
            raise ValueError(message)
         
    def _check_values_not_none(self):
        r"""
        Checks if the ``values`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``values`` is ``None``. 
        """
        if self.values is None:
            message = (
                "``values`` cannot be None! You can load the instances using the methods"
                "``from_data`` or ``from_txt``."
            )
            raise ValueError(message)
    
    def _check_capacities_not_none(self):
        r"""
        Checks if the ``capacities`` attribute is not ``None``. 
        Raises a ``ValueError`` if ``capacities`` is ``None``. 
        """
        if self.capacities is None:
            message = (
                "``capacities`` cannot be None! You can load the instances using the methods"
                "``from_data`` or ``from_txt``."
            )
            raise ValueError(message)
    
    def _check_items_label_not_none(self, ref: bool):
        r"""
        Checks if the ``items_label` or ``ref_items_label`` attribute is not ``None``.
        - If ``ref`` is ``True``, it checks the ``ref_items_label`` attribute.
        - If ``ref`` is ``False``, it checks the ``items_label`` attribute.
        Raises a `ValueError` if the respective attribute is ``None``.
        """
        msg = "ref_items_label" if ref else "items_label"
        message = (
            f"``{msg}`` cannot be None! You can use solvers based on "
            "``KPSolver`` like ``KPORSolver`` or use methods including "
            "``from_data``or ``from_txt`` to obtain them."
        )  
        if ref:
            if self.ref_items_label is None:
                raise ValueError(message)
        else:
            if self.items_label is None:    
                raise ValueError(message)
    
    def _check_capacities_meet(self):
        r"""
        Checks if the ``items_label`` satisfies the capacity limits. Raise a `ValueError` 
        if there is a selection of items that exceed the capacity.
        """
        items_label_shape = self.items_label.shape
        for idx in range(items_label_shape[0]):
            cur_capacity = self.capacities[idx]
            cur_weights_total = (self.items_label[idx] * self.weights[idx]).sum()
            
            if cur_weights_total > cur_capacity + 1e-5:
                message = (
                    f"Capacity constraint not met for instance {idx}. "
                    f"The selection is ``{self.items_label[idx]}`` with the total weights of {cur_weights_total}."
                    f"However, the maximum capacity of the knapsack is {cur_capacity}."
                )
                raise ValueError(message)

    def _apply_scale_and_dtype(
        self, 
        weights: np.ndarray,
        values: np.ndarray,
        capacities: np.ndarray,
    ):
        r"""
        This function scales the given ``weights``, ``values``, and ``capacities`` by a factor of ``self.scale``
        for solvers like OR-Tools that accepets only integer inputs for the KP instances.
        """
        # apply scale
        weights = (weights * self.scale).astype(np.int32)
        values = (values * self.scale).astype(np.int32)
        capacities = (capacities * self.scale).astype(np.int32)
        
        return weights, values, capacities

    def from_txt(
        self,
        file_path: str,
        ref: bool = False,
        return_list: bool = False,
        show_time: bool = False
    ):
        """
        Read data from `.txt` file.

        :param file_path: string, path to the `.txt` file containing CVPR instances data.
        :param ref: boolean, whether the solution is a reference solution.
        :param return_list: boolean, only use this function to obtain data, but do not save it to the solver. 
        :param show_time: boolean, whether the data is being read with a visual progress display.

        .. dropdown:: Example
        
            ::

                >>> from ml4co_kit import KPSolver
                
                # create KPSolver
                >>> solver = KPSolver()

                # load data from ``.txt`` file
                >>> solver.from_txt(file_path="examples/kp/txt/kp100.txt")
                >>> solver.items_label.shape
                (16, 100)
                >>> solver.weights.shape
                (16, 100)
                >>> solver.capacities.shape
                (16,)
        """
        # check the file format
        if not file_path.endswith(".txt"):
            raise ValueError("Invalid file format. Expected a ``.txt`` file.")

        # read the data form .txt
        with open(file_path, "r") as file:
            # record to lists
            weights_list = list()
            values_list = list()
            capacities_list = list()
            items_label_list = list()
            
            # read by lines
            for line in iterative_execution_for_file(file, "Loading", show_time):
                # line to strings
                line = line.strip()
                split_line_0 = line.split("weights ")[1]
                split_line_1 = split_line_0.split(" values ")
                weights = split_line_1[0]
                split_line_2 = split_line_1[1].split(" capacity ")
                values = split_line_2[0]
                split_line_3 = split_line_2[1].split(" label ")
                capacity = split_line_3[0]
                items_label = split_line_3[1]
                
                # strings to array
                weights = weights.split(" ")
                weights = np.array([
                    float(weights[i]) for i in range(len(weights))
                ])
                values = values.split(" ")
                values = np.array([
                    float(values[i]) for i in range(len(values))
                ])
                items_label = items_label.split(" ")
                items_label = np.array([
                    int(items_label[i]) for i in range(len(items_label))
                ])
                capacity = float(capacity)
                
                # add to the list
                weights_list.append(weights)
                values_list.append(values)
                capacities_list.append(capacity)
                items_label_list.append(items_label)

        # check if return list
        if return_list:
            return weights_list, values_list, capacities_list, items_label_list
        
        # use ``from_data``
        self.from_data(
            weights=weights_list, values=values_list, 
            capacities=capacities_list,
            items_label=items_label_list, ref=ref
        )
              
    def from_data(
        self,
        weights: Union[list, np.ndarray] = None,
        values: Union[list, np.ndarray] = None,
        capacities: Union[int, float, np.ndarray] = None,
        items_label: Union[list, np.ndarray] = None,
        ref: bool = False,
    ):
        """
        Read data from list or np.ndarray.

        :param weights: np.ndarray, the weights of items. If given, the weights 
            originally stored in the solver will be replaced.
        :param values: np.ndarray, the values of items. If given, the values 
            originally stored in the solver will be replaced.
        :param capacities: int, float or np.ndarray, the capacities of the knapsack. If given, the capacities 
            originally stored in the solver will be replaced.
        :param items_label: np.ndarray, the solutions of the problems. If given, the solution labels
            originally stored in the solver will be replaced
        :param ref: boolean, whether the solution is a reference solution.

        .. dropdown:: Example

            :: 

                >>> import numpy as np
                >>> from ml4co_kit import KPSolver
                
                # create KPSolver
                >>> solver = KPSolver()

                # load data from np.ndarray
                >>> solver.from_data(
                        weights=np.random.random(size=50),
                        values=np.random.random(size=50),
                        capacities=np.random.random(size=1)
                    )
                >>> solver.weights.shape
                (1, 50)
        """
        
        # weights
        if weights is not None:
            weights = to_numpy(weights)
            self.weights = weights.astype(np.float32)
            self._check_weights_dim()
        
        # values
        if values is not None:
            values = to_numpy(values)
            self.values = values.astype(np.float32)
            self._check_values_dim()
        
        # capacities
        if capacities is not None:
            if isinstance(capacities, (float, int)):
                capacities = np.array([capacities]).astype(np.float32)
            if isinstance(capacities, list):
                capacities = np.array(capacities)
            self.capacities = capacities.astype(np.float32)
            self._check_capacities_dim()

        # items_label (ref or not)
        if items_label is not None:
            if isinstance(items_label, list):
                # 1D labels
                if not isinstance(items_label[0], list) and not isinstance(items_label[0], np.ndarray):
                    items_label = np.array(items_label)
                # 2D labels
                else:
                    items_label = to_numpy(items_label).astype(np.int32)
            if ref:
                self.ref_items_label = items_label
                self._check_ref_items_label_dim()
            else:
                self.items_label = items_label
                self._check_items_label_dim()
                
    def to_txt(
        self,
        file_path: str = "example.txt",
        apply_scale: bool = False
    ):
        """
        Output(store) data in ``txt`` format

        :param file_path: string, path to save the `.txt` file.
        :param apply_scale: boolean, whether to scale the ``weights``, ``values``, and ``capacities`` 
            by a factor of ``self.scale`` (often to integers).

        .. note::
            ``weights``, ``values``,``capacities`` and ``items_label`` must not be None.
         
        .. dropdown:: Example

            :: 
            
                >>> from ml4co_kit import KPSolver
                >>> import numpy as np
                
                # create KPORSolver
                >>> solver = KPORSolver()

                # generate data randomly
                >>> solver.from_data(
                        weights=np.random.random(size=(16, 50)),
                        values=np.random.random(size=(16, 50)),
                        capacities=[10 for _ in range(16)],
                        ref=False,
                    )

                # solve
                >>> solver.solve()
                    
                # Output data in ``txt`` format
                >>> solver.to_txt("KP50_example.txt")
        """
        # check
        self._check_weights_not_none()
        self._check_values_not_none()
        self._check_capacities_not_none()
        self._check_items_label_not_none(ref=False)
        
        # variables
        weights = self.weights
        values = self.values
        capacities = self.capacities
        items_label = self.items_label
        
        # apply scale and dtype
        if apply_scale:
            weights, values, capacities = self._apply_scale_and_dtype(
                weights=weights, values=values, capacities=capacities
            )
        
        # write
        with open(file_path, "w") as f:
            for idx, vars in enumerate(items_label):
                capacity = self.capacities[idx]
                weights = self.weights[idx]
                values = self.values[idx]
                f.write("weights " + str(" ").join(str(weight) for weight in weights))
                f.write(" values " + str(" ").join(str(value) for value in values))
                f.write(" capacity " + str(capacity))
                f.write(str(" label "))
                f.write(str(" ").join(str(var) for var in vars))
                f.write("\n")

    def evaluate(
        self,
        calculate_gap: bool = False,
        check_capacity: bool = True,
        apply_scale: bool = False,
    ):
        """
        Evaluate the solution quality of the solver

        :param calculate_gap: boolean, whether to calculate the gap with the reference solutions.
        :param _check_capacity: boolean, whether to check if demands are met.
        :param apply_scale: boolean, whether to scale and convert the data back to floats for reporting.

        .. note::
            - Please make sure the ``values`` and the ``items_label`` are not None.
            - If you set the ``calculate_gap`` as True, please make sure the ``ref_items_label`` is not None.
        
        .. dropdown:: Example

            :: 
            
                >>> from ml4co_kit import KPSolver
                
                # create KPSolver
                >>> solver = KPSolver()

                # load data and reference solutions from ``.txt`` file
                >>> solver.from_txt(file_path="examples/kp/txt/kp100.txt", ref=True)
                >>> sol_label = solver.ref_items_label.copy()
                >>> sol_label[:, 0] = 0
                >>> solver.items_label = sol_label
                    
                # Evaluate the quality of the solutions with manual perturbations
                >>> solver.evaluate(calculate_gap=True)
                (41.093995961913606, 41.7070689011307, 1.4679890697337488, 0.8475666327662943)
        """
        # check
        self._check_values_not_none()
        self._check_items_label_not_none(ref=False)
        if check_capacity:
            self._check_capacities_not_none()
            self._check_weights_not_none()
            self._check_capacities_meet()
        if calculate_gap:
            self._check_items_label_not_none(ref=True)
            
        # variables
        weights = self.weights
        values = self.values
        capacities = self.capacities
        items_label = self.items_label
        ref_items_label = self.ref_items_label

        # apply scale and dtype
        if apply_scale:
            weights, values, capacities = self._apply_scale_and_dtype(
                weights=weights, values=values, capacities=capacities
            )
        
        # prepare for evaluate
        sol_values_list = list()
        instance_num = values.shape[0]
        if calculate_gap:
            ref_sol_values_list = list()
            gap_list = list()
            
        # evaluate
        for idx in range(instance_num):
            sol_value = (items_label[idx] * values[idx]).sum()
            sol_values_list.append(sol_value)
            if calculate_gap:
                ref_value = (ref_items_label[idx] * values[idx]).sum()
                ref_sol_values_list.append(ref_value)
                gap = (ref_value - sol_value) / ref_value * 100
                gap_list.append(gap)

        # calculate average cost/gap & std
        sol_values = np.array(sol_values_list)
        if calculate_gap:
            ref_sol_values = np.array(ref_sol_values_list)
            gaps = np.array(gap_list)
        values_avg = np.average(sol_values)
        if calculate_gap:
            ref_values_avg = np.average(ref_sol_values)
            gap_avg = np.sum(gaps) / instance_num
            gap_std = np.std(gaps)
            return values_avg, ref_values_avg, gap_avg, gap_std
        else:
            return values_avg
        
    def solve(
        self,
        weights: Union[list, np.ndarray] = None,
        values: Union[list, np.ndarray] = None,
        capacities: Union[list, np.ndarray] = None,
        num_threads: int = 1,
        **kwargs,
    ) -> np.ndarray:
        """
        This method will be implemented in subclasses.
        
        :param weights: np.ndarray, the item weights data called by the solver during solving.
        :param values: np.ndarray, the item values data called by the solver during solving.
        :param capacities: np.ndarray, the capacities of the knapsack(s).
        :param num_threads: int, number of threads(could also be processes) used in parallel.
        :param show_time: boolean, whether the data is being read with a visual progress display.

        .. dropdown:: Example

            ::
            
                >>> from ml4co_kit import KPORSolver
                
                # create KPORSolver
                >>> solver = KPORSolver()

                # load data and reference solutions from ``.txt`` file
                >>> solver.from_txt(file_path="examples/kp/txt/kp100.txt", ref=False)
                    
                # solve
                >>> solver.solve()
                (41.7070689011307, 41.7070689011307, 0.0, 0.0)
        """
        raise NotImplementedError(
            "The ``solve`` function is required to implemented in subclasses."
        )

    def __str__(self) -> str:
        return "KPSolver"