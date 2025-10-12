r"""
Base class for all wrappers in the ML4CO kit.
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


import os
import random
import pickle
import pathlib
import numpy as np
from typing import Sequence, Union, Type
from multiprocessing import Pool
from ml4co_kit.solver.base import SolverBase
from ml4co_kit.utils.time_utils import Timer
from ml4co_kit.generator.base import GeneratorBase
from ml4co_kit.utils.time_utils import tqdm_by_time
from ml4co_kit.task.base import TASK_TYPE, TaskBase
from ml4co_kit.utils.file_utils import check_file_path


class WrapperBase(object):
    def __init__(
        self,
        task_type: TASK_TYPE, 
        precision: Union[np.float32, np.float64] = np.float32
    ):
        self.task_type = task_type
        self.precision = precision
        self.task_list = list()
    
    def swap_sol_and_ref_sol(self):
        for task_data in self.task_list:
            tmp = task_data.ref_sol
            task_data.ref_sol = task_data.sol
            task_data.sol = tmp
 
    def generate_w_to_txt(
        self,
        file_path: pathlib.Path,
        generator: GeneratorBase,
        solver: SolverBase,
        num_samples: int = 1280,
        num_threads: int = 1,
        batch_size: int = 1,
        write_per_iters: int = 1,
        write_mode: str = "a",
        show_time: bool = True
    ):
        # Calculate the total number of iterations
        if num_samples % (num_threads * batch_size * write_per_iters) != 0:
            raise ValueError((
                "The number of samples must be divisible by "
                "the product of num_threads, batch size, and write_per_iters."
            ))
        tot_iters = num_samples // num_threads // batch_size
        
        # Generate tasks by chunks
        for cur_iter in tqdm_by_time(
            iterable=range(tot_iters),
            desc=f"Generating {self.task_type}",
            show_time=show_time
        ):
            # Generate tasks
            self.generate(
                generator=generator,
                solver=solver,
                num_samples=num_threads*batch_size,
                num_threads=num_threads,
                batch_size=batch_size,
                show_time=False
            )
            
            # Write tasks to txt
            if (cur_iter+1) % write_per_iters == 0:
                self.to_txt(file_path=file_path, show_time=False, mode=write_mode)
                self.task_list = list()
                
    def generate(
        self, 
        generator: GeneratorBase,
        solver: SolverBase,
        num_samples: int = 1280,
        num_threads: int = 1,
        batch_size: int = 1,
        show_time: bool = True
    ):
        # Initialize Timer
        timer = Timer(apply=show_time)
        timer.start()
        
        # Case 1: Single Thread and Batch Size is 1
        if num_threads == 1 and batch_size == 1:
            for _ in tqdm_by_time(
                iterable=range(num_samples), 
                desc=f"Generating {self.task_type}", 
                show_time=show_time
            ):
                task_data = generator.generate()
                solver.solve(task_data)
                self.task_list.append(task_data)
                
        # Case 2: Multi Thread and Batch Size is 1 (usually for traditional solver, cpu)
        elif num_threads != 1 and batch_size == 1:
            # Check if the number of samples is divisible by the number of threads
            if num_samples % num_threads != 0:
                raise ValueError(
                    "The number of samples must be divisible by the number of threads."
                )
            # Generate Tasks
            for _ in tqdm_by_time(
                iterable=range(num_samples // num_threads), 
                desc=f"Generating {self.task_type}", 
                show_time=show_time
            ):
                with Pool(num_threads) as p1:
                    tasks = p1.starmap(
                        self._generate, 
                        [(generator, solver) for _ in range(num_threads)]
                    )
                    self.task_list.extend(tasks)
                    
        # Case 3: Single Thread and Batch Size is not 1 (usually for ML4CO solver, gpu)
        elif num_threads == 1 and batch_size != 1:
            # Check if the number of samples is divisible by the batch size
            if num_samples % batch_size != 0:
                raise ValueError(
                    "The number of samples must be divisible by the batch size."
                )
            # Generate Tasks
            for _ in tqdm_by_time(
                iterable=range(num_samples // batch_size), 
                desc=f"Generating {self.task_type}", 
                show_time=show_time
            ):
                batch_task_data = [generator.generate() for _ in range(batch_size)]
                solver.batch_solve(batch_task_data)
                self.task_list.extend(batch_task_data)
                
        # Case 4: Multi Thread and Batch Size is not 1
        else:
            raise ValueError((
                "``num_threads`` and ``batch_size`` cannot "
                "both be greater than 1 at the same time."
            ))

        # End Timer
        timer.end()
        timer.show_time()

    def _generate(self, generator: GeneratorBase, solver: SolverBase) -> TaskBase:
        seed = os.getpid() % 2**32          # Seed
        random.seed(seed)                   # Set seed for random
        np.random.seed(seed)                # Set seed for numpy
        task_data = generator.generate()    # Generate Task Data
        solver.solve(task_data)             # Solve Task Data
        return task_data
    
    def from_txt(self, file_path: pathlib.Path, *args, **kwargs):
        raise NotImplementedError(
            "The ``from_txt`` function is required to implemented in subclasses."
        )
        
    def to_txt(file_path: pathlib.Path, show_time: bool = False, mode: str = "w"):
        raise NotImplementedError(
            "The ``to_txt`` function is required to implemented in subclasses."
        )
    
    def from_pickle(self, file_path: pathlib.Path):
        with open(file_path, "rb") as file:
            self.task_list = pickle.load(file)
        
    def to_pickle(self, file_path: pathlib.Path):
        # Check file path
        check_file_path(file_path)
        
        # Save task data to ``.pkl`` file
        with open(file_path, "wb") as f:
            pickle.dump(self.task_list, f)
            f.close()
    
    def from_task_pickle_folder(
        self, task_class: Type[TaskBase], folder_path: pathlib.Path
    ):
        # Get pickle files
        pickle_files = os.listdir(folder_path)
        pickle_files.sort()
        
        # Load task data
        self.task_list = list()
        for pickle_file in pickle_files:
            pkl_path = folder_path / pickle_file
            task_data = task_class()
            task_data.from_pickle(pkl_path)
            self.task_list.append(task_data)
    
    def to_task_pickle_folder(self, folder_path: pathlib.Path):
        # Create folder
        os.makedirs(folder_path, exist_ok=True)
        
        # Save task data
        for task_data in self.task_list:
            pkl_path = folder_path / f"{task_data.name}.pkl"
            task_data.to_pickle(pkl_path)
    
    def solve(
        self, 
        solver: SolverBase, 
        num_threads: int = 1, 
        batch_size: int = 1,
        show_time: bool = False
    ):
        # Initialize Timer
        timer = Timer(apply=show_time)
        timer.start()
        
        # Solving Message
        solve_msg = f"Solving {self.task_type.value} Using {solver.solver_type.value}"
        
        # Case 1: Single Thread and Batch Size is 1
        if num_threads == 1 and batch_size == 1:
            for task_data in tqdm_by_time(
                iterable=self.task_list,
                desc=solve_msg,
                show_time=show_time
            ):
                solver.solve(task_data)
                
        # Case 2: Multi Thread and Batch Size is 1
        elif num_threads != 1 and batch_size == 1:
            # Check if the number of tasks is divisible by the number of threads
            if len(self.task_list) % num_threads != 0:
                raise ValueError(
                    "The number of tasks must be divisible by the number of threads."
                )
            # Solve Tasks
            for idx in tqdm_by_time(
                iterable=range(len(self.task_list) // num_threads), 
                desc=solve_msg,
                show_time=show_time
            ):
                with Pool(num_threads) as p1:
                    task_data_list = p1.map(
                        solver.solve, 
                        [self.task_list[idx*num_threads+i] for i in range(num_threads)]
                    )
                    for j, task_data in enumerate(task_data_list):
                        self.task_list[idx*num_threads+j] = task_data
                        
        # Case 3: Single Thread and Batch Size is not 1
        elif num_threads == 1 and batch_size != 1:
            # Check if the number of tasks is divisible by the batch size
            if len(self.task_list) % batch_size != 0:
                raise ValueError(
                    "The number of tasks must be divisible by the batch size."
                )
            # Solve Tasks
            for idx in tqdm_by_time(
                iterable=range(len(self.task_list) // batch_size), 
                desc=solve_msg,
                show_time=show_time
            ):
                batch_task_data = [self.task_list[idx*batch_size+i] for i in range(batch_size)]
                solver.batch_solve(batch_task_data)
                
        # Case 4: Multi Thread and Batch Size is not 1
        else:
            raise ValueError((
                "``num_threads`` and ``batch_size`` cannot "
                "both be greater than 1 at the same time."
            ))
        
        # End Timer
        timer.end()
        timer.show_time()
    
    def evaluate(self) -> float:
        """Evaluate the task list."""
        sol_costs_list = list()
        for task_data in self.task_list:
            sol_cost = task_data.evaluate(task_data.sol)
            sol_costs_list.append(sol_cost)
        return float(np.mean(sol_costs_list))
    
    def evaluate_w_gap(self) -> Sequence[float]:
        """Evaluate the task list."""
        # Initialize lists
        sol_costs_list = list()
        ref_costs_list = list()
        gaps_list = list()
        
        # Evaluate the task list
        for task_data in self.task_list:
            sol_cost, ref_cost, gap = task_data.evaluate_w_gap()
            sol_costs_list.append(sol_cost)
            ref_costs_list.append(ref_cost)
            gaps_list.append(gap)
        
        # Calculate the average solution cost, reference cost, and gap
        avg_sol_cost = float(np.mean(sol_costs_list))
        avg_ref_cost = float(np.mean(ref_costs_list))
        if None not in gaps_list:
            avg_gap = float(np.mean(gaps_list))
            gap_std = float(np.std(gaps_list))
        else:
            avg_gap = None
            gap_std = None
        return avg_sol_cost, avg_ref_cost, avg_gap, gap_std