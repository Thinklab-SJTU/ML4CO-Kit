import copy
import shutil
import pathlib
import numpy as np
from typing import Union, Tuple
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.generator.base import EdgeGeneratorBase
from ml4co_kit.solver import ATSPSolver, ATSPLKHSolver


class ATSPDataGenerator(EdgeGeneratorBase):
    def __init__(
        self,
        only_instance_for_us: bool = False,
        num_threads: int = 1,
        nodes_num: int = 50,
        data_type: str = "uniform",
        solver: Union[SOLVER_TYPE, ATSPSolver, str] = SOLVER_TYPE.LKH,
        train_samples_num: int = 128000,
        val_samples_num: int = 1280,
        test_samples_num: int = 1280,
        save_path: pathlib.Path = "dataset/atsp",
        filename: str = None,
        # special for sat
        sat_vars_nums: int = 5,
        sat_clauses_nums: int = 5,
    ):
        # filename
        if data_type == "sat":
            nodes_num = 2 * sat_clauses_nums * sat_vars_nums + sat_clauses_nums
        if filename is None:
            filename = f"atsp{nodes_num}_{data_type}"

        # special args
        self.sat_vars_nums = sat_vars_nums
        self.sat_clauses_nums = sat_clauses_nums

        # re-define
        generate_func_dict = {
            "sat": self._generate_sat,
            "hcp": self._generate_hcp,
            "uniform": self._generate_uniform
        }
        supported_solver_dict = {
            SOLVER_TYPE.LKH: ATSPLKHSolver
        }
        check_solver_dict = {
            SOLVER_TYPE.LKH: self._check_lkh
        }

        # super args
        super(ATSPDataGenerator, self).__init__(
            only_instance_for_us=only_instance_for_us,
            num_threads=num_threads,
            nodes_num=nodes_num,
            data_type=data_type,
            solver=solver,
            train_samples_num=train_samples_num,
            val_samples_num=val_samples_num,
            test_samples_num=test_samples_num,
            save_path=save_path,
            filename=filename,
            generate_func_dict=generate_func_dict,
            supported_solver_dict=supported_solver_dict,
            check_solver_dict=check_solver_dict
        )
        self.solver: ATSPSolver

    ##################################
    #         Generate Funcs         #
    ##################################
    
    def _generate_sat(self) -> Tuple[np.ndarray, np.ndarray]:
        dists, ref_tours = [], []
        num_nodes = self.nodes_num
        num_variables = self.sat_vars_nums
        num_clauses = self.sat_clauses_nums
        
        # Generate SAT instances for each thread
        for _ in range(self.num_threads):
            # Initialize distance matrix with ones
            dist = np.ones((num_nodes, num_nodes))
            ref_tour = []
            # Randomly generate variable values (0 or 1)
            var_values = [np.random.randint(0, 2) for _ in range(num_variables)]
            
            # Generate the distance matrix and reference tour
            for v in range(num_variables):
                sub_tour = []
                ofs = v * 2 * num_clauses  # Offset for variable nodes
                
                for c in range(num_clauses):
                    # Set the distances between clause pairs to 0
                    dist[ofs + 2 * c, ofs + 2 * c + 1] = 0
                    dist[ofs + 2 * c + 1, ofs + 2 * c] = 0
                    
                    # Build the sub-tour based on variable value
                    if var_values[v] == 1:
                        sub_tour.append(ofs + 2 * c + 1)
                        if c != num_clauses - 1:
                            sub_tour.append(ofs + 2 * c + 2)
                    else:
                        sub_tour.insert(0, ofs + 2 * c)
                        if c != num_clauses - 1:
                            sub_tour.insert(0, ofs + 2 * c + 1)
                        
                    # Connect clauses in sequence
                    if c != num_clauses - 1:
                        dist[ofs + 2 * c + 1, ofs + 2 * c + 2] = 0
                        dist[ofs + 2 * c + 2, ofs + 2 * c + 1] = 0

                # Connect variable clauses to the next variable
                dist[ofs, (ofs + 2 * num_clauses) % (2 * num_variables * num_clauses)] = 0
                dist[ofs, (ofs + 4 * num_clauses - 1) % (2 * num_variables * num_clauses)] = 0
                dist[ofs + 2 * num_clauses - 1, (ofs + 2 * num_clauses) % (2 * num_variables * num_clauses)] = 0
                dist[ofs + 2 * num_clauses - 1, (ofs + 4 * num_clauses - 1) % (2 * num_variables * num_clauses)] = 0

                # Extend the reference tour with the sub-tour
                ref_tour.extend(sub_tour)
                if var_values[(v + 1) % num_variables] == 1:
                    ref_tour.append((ofs + 2 * num_clauses) % (2 * num_variables * num_clauses))
                else:
                    ref_tour.append((ofs + 4 * num_clauses - 1) % (2 * num_variables * num_clauses))

            # Close the tour by connecting the last node to the first
            ref_tour.insert(0, ref_tour[-1])
            ofs_clause = 2 * num_clauses * num_variables

            for c in range(num_clauses):
                # Randomly select 3 variables for the clause
                vars = np.random.choice(num_variables, size=3, replace=False)
                # Randomly assign signs (0 or 1) to the selected variables
                signs = np.random.choice(2, 3, replace=True)
                # Ensure at least one variable satisfies the clause
                fix_var_id = np.random.randint(0, 3)
                signs[fix_var_id] = var_values[vars[fix_var_id]]
                
                # Update distance matrix and reference tour for the clause
                for i in range(3):
                    ofs_var = vars[i] * 2 * num_clauses
                    if signs[i] == 1:  # Variable is True
                        dist[ofs_var + 2 * c, ofs_clause + c] = 0
                        dist[ofs_clause + c, ofs_var + 2 * c + 1] = 0
                        if vars[i] == vars[fix_var_id]:
                            ref_tour.insert(ref_tour.index(ofs_var + 2 * c + 1), ofs_clause + c)
                    else:  # Variable is False (not x)
                        dist[ofs_var + 2 * c + 1, ofs_clause + c] = 0
                        dist[ofs_clause + c, ofs_var + 2 * c] = 0
                        if vars[i] == vars[fix_var_id]:
                            ref_tour.insert(ref_tour.index(ofs_var + 2 * c), ofs_clause + c)
                            
            # Append the generated distance matrix and reference tour to the lists
            dists.append(dist)
            ref_tours.append(ref_tour)
            
        # Return the distance matrices and reference tours as numpy arrays
        return np.array(dists), np.array(ref_tours)

    def _generate_hcp(self) -> Tuple[np.ndarray, np.ndarray]:
        # prepare
        dists, ref_tours = [], []
        noise_level = np.random.rand() * 0.2 + 0.1
        num_nodes = self.nodes_num
        
        # generate
        for _ in range(self.num_threads):
            # Ones matrix
            dist = np.ones((num_nodes, num_nodes))

            # Random permutation of nodes (equivalent to torch.randperm)
            hpath = np.random.permutation(num_nodes)

            # Set distances to 0 along the hpath
            dist[hpath, np.roll(hpath, -1)] = 0

            # Add noise to the distance matrix
            num_noise_edges = int(noise_level * num_nodes * num_nodes)
            if num_noise_edges > 0:
                heads = np.random.choice(num_nodes, size=num_noise_edges, replace=True)
                tails = np.random.choice(num_nodes, size=num_noise_edges, replace=True)
                dist[heads, tails] = 0

            # Convert the hpath to a list and append the first node to form a closed tour
            ref_tour: list = hpath.tolist()
            ref_tour.append(ref_tour[0])
            
            dists.append(dist)
            ref_tours.append(ref_tour)
        
        return np.array(dists), np.array(ref_tours)

    def _generate_uniform(self) -> Tuple[np.ndarray, np.ndarray]:
        scaler = 1e6
        dists = list()
        for _ in range(self.num_threads):
            dist = np.random.randint(low=0, high=scaler, size=(self.nodes_num, self.nodes_num))
            dist[np.arange(self.nodes_num), np.arange(self.nodes_num)] = 0
            dist: np.ndarray
            while True:
                old_dist = copy.deepcopy(dist)
                dist = (dist[:, None, :] + dist[None, :, :].transpose(0, 2, 1)).min(axis=2)
                if (dist == old_dist).all():
                    break
            dists.append(dist / scaler)
        return np.array(dists), None

    ##################################
    #      Solver-Checking Funcs     #
    ##################################
    
    def _check_lkh(self):
        # check if lkh is downloaded
        if shutil.which(self.solver.lkh_path) is None:
            self.download_lkh()
        # check again
        if shutil.which(self.solver.lkh_path) is None:
            message = (
                f"The LKH solver cannot be found in the path '{self.solver.lkh_path}'. "
                "Please make sure that you have entered the correct ``lkh_path``."
                "If you have not installed the LKH solver, "
                "please use function ``self.download_lkh()`` to download it."
                "Please also confirm whether the Conda environment of the terminal "
                "is consistent with the Python environment."
            )
            raise ValueError(message)

    ##################################
    #      Data-Generating Funcs     #
    ##################################
    
    def generate_only_instance_for_us(self, samples: int) -> np.ndarray:
        self.num_threads = samples
        dists = self.generate_func()[0]
        self.solver.from_data(dists=dists)
        return self.solver.dists

    def _generate_core(self):
        # call generate_func to generate data
        batch_dists, tours = self.generate_func()
        
        # solve
        if tours is None:
            tours = self.solver.solve(
                dists=batch_dists, num_threads=self.num_threads
            )
        
        # write to txt
        for idx, tour in enumerate(tours):
            tour = tour[:-1]
            if (np.sort(tour) == np.arange(self.nodes_num)).all():
                with open(self.file_save_path, "a+") as f:
                    for line in batch_dists[idx]:
                        f.write(" ".join(str(x) + str(" ") for x in line))
                        f.write(str(" "))
                    f.write(str("output") + str(" "))
                    f.write(str(" ").join(str(node_idx + 1) for node_idx in tour))
                    f.write(str(" ") + str(tour[0] + 1) + str(" "))
                    f.write("\n")
                f.close()