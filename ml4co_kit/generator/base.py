import os
import sys
import time
import random
import shutil
import pathlib
import numpy as np
import networkx as nx
from tqdm import tqdm
from typing import Union, Any
from ml4co_kit.solver.base import SolverBase
from ml4co_kit.utils.type_utils import SOLVER_TYPE


import warnings
warnings.filterwarnings("ignore")

class GeneratorBase(object):
    def __init__(
        self,
        only_instance_for_us: bool,
        num_threads: int,
        data_type: str,
        solver: Union[SOLVER_TYPE, SolverBase, str],
        train_samples_num: int,
        val_samples_num: int,
        test_samples_num: int,
        save_path: pathlib.Path,
        filename: str,
        generate_func_dict: dict,
        supported_solver_dict: dict,
        check_solver_dict: dict
    ) -> None:
        # basic params
        self.only_instance_for_us = only_instance_for_us
        self.num_threads = num_threads
        self.data_type = data_type
        self.solver = solver
        self.train_samples_num = train_samples_num
        self.val_samples_num = val_samples_num
        self.test_samples_num = test_samples_num
        self.save_path = save_path
        self.filename = filename
        
        # check the data type
        self.generate_func_dict = generate_func_dict
        self._check_data_type()

        # check the solver 
        self.supported_solver_dict = supported_solver_dict
        self.check_solver_dict = check_solver_dict
        self._check_solver()
 
        # other checks
        if not only_instance_for_us:
            # 1. check the multi-threads
            self.sample_types = ["train", "val", "test"]
            self._check_num_threads()   
            
            # 2. check the save path
            self._get_filename()

    def _check_data_type(self):
        supported_data_type = self.generate_func_dict.keys()
        if self.data_type not in supported_data_type:
            message = (
                f"The input data_type ({self.data_type}) is not a valid type, "
                f"and the generator only supports {supported_data_type}."
            )
            raise ValueError(message)
        self.generate_func = self.generate_func_dict[self.data_type]
    
    def _check_num_threads(self):
        self.samples_num = 0
        for sample_type in self.sample_types:
            self.samples_num += getattr(self, f"{sample_type}_samples_num")
            if self.samples_num % self.num_threads != 0:
                message = "``samples_num`` must be divisible by the number of threads. "
                raise ValueError(message)

    def _check_solver(self):
        # get solver
        if isinstance(self.solver, (SOLVER_TYPE, str)):
            self.solver_type = self.solver
            supported_solver_type = self.supported_solver_dict.keys()
            if self.solver_type not in supported_solver_type:
                message = (
                    f"The input solver_type ({self.solver_type}) is not a valid type, "
                    f"and the generator only supports {supported_solver_type}."
                )
                raise ValueError(message)
            self.solver = self.supported_solver_dict[self.solver_type]()
        else:
            self.solver: SolverBase
            self.solver_type = self.solver.solver_type
            
        # check solver
        check_func = self.check_solver_dict[self.solver_type]
        check_func()
        
    def _get_filename(self):
        self.file_save_path = os.path.join(self.save_path, self.filename + ".txt")
        for sample_type in self.sample_types:
            setattr(
                self,
                f"{sample_type}_file_save_path",
                os.path.join(self.save_path, self.filename + f"_{sample_type}.txt"),
            )
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
    
    def _check_free(self):
        return
            
    def generate_only_instance_for_us(self, samples: int) -> Any:
        raise NotImplementedError(
            "The ``generate_only_instance_for_us`` function is required to implemented in subclasses."
        )

    def generate(self) -> Any:
        start_time = time.time()
        for _ in tqdm(
            range(self.samples_num // self.num_threads),
            desc=self.solver.solve_msg,
        ):
            self._generate_core()
        end_time = time.time() - start_time
        print(
            f"Completed generation of {self.samples_num} samples of {self.solver.task_type}."
        )
        print(f"Total time: {end_time/60:.1f}m")
        print(f"Average time: {end_time/self.samples_num:.1f}s")
        self.devide_file()
     
    def _generate_core(self) -> Any:
        raise NotImplementedError(
            "The ``_generate_core`` function is required to implemented in subclasses."
        )
   
    def devide_file(self):
        with open(self.file_save_path, "r") as f:
            data = f.readlines()
        train_end_idx = self.train_samples_num
        val_end_idx = self.train_samples_num + self.val_samples_num
        train_data = data[:train_end_idx]
        val_data = data[train_end_idx:val_end_idx]
        test_data = data[val_end_idx:]
        data = [train_data, val_data, test_data]
        for sample_type, data_content in zip(self.sample_types, data):
            filename = getattr(self, f"{sample_type}_file_save_path")
            with open(filename, "w") as file:
                file.writelines(data_content)


class EdgeGeneratorBase(GeneratorBase):
    def __init__(
        self,
        only_instance_for_us: bool,
        num_threads: int,
        nodes_num: int,
        data_type: str,
        solver: Union[SOLVER_TYPE, SolverBase],
        train_samples_num: int,
        val_samples_num: int,
        test_samples_num: int,
        save_path: pathlib.Path,
        filename: str,
        # need to pre-define
        generate_func_dict: dict,
        supported_solver_dict: dict,
        check_solver_dict: dict
    ):
        # super args
        super(EdgeGeneratorBase, self).__init__(
            only_instance_for_us=only_instance_for_us,
            num_threads=num_threads,
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
        
        # other args
        self.nodes_num = nodes_num
        
    def download_lkh(self):
        # download
        import wget
        lkh_url = "http://akira.ruc.dk/~keld/research/LKH-3/LKH-3.0.7.tgz"
        wget.download(url=lkh_url, out="LKH-3.0.7.tgz")
        # tar .tgz file
        os.system("tar xvfz LKH-3.0.7.tgz")
        # build LKH
        ori_dir = os.getcwd()
        os.chdir("LKH-3.0.7")
        os.system("make")
        # move LKH to the bin dir
        target_dir = os.path.join(sys.prefix, "bin")
        os.system(f"cp LKH {target_dir}")
        os.chdir(ori_dir)
        # delete .tgz file
        os.remove("LKH-3.0.7.tgz")
        shutil.rmtree("LKH-3.0.7")
        
        
class NodeGeneratorBase(GeneratorBase):
    def __init__(
        self,
        only_instance_for_us: bool,
        num_threads: int,
        data_type: str,
        solver: Union[SOLVER_TYPE, SolverBase],
        train_samples_num: int,
        val_samples_num: int,
        test_samples_num: int,
        save_path: pathlib.Path,
        filename: str,
        # args for generate
        nodes_num_min: int,
        nodes_num_max: int,
        graph_weighted: bool,
        er_prob: float,
        ba_conn_degree: int,
        hk_prob: float,
        hk_conn_degree: int,
        ws_prob: float,
        ws_ring_neighbors: int,
        # need to pre-define
        supported_solver_dict: dict,
        check_solver_dict: dict
    ):
        # generate funcs
        generate_func_dict = {
            "erdos_renyi": self._generate_erdos_renyi,
            "er": self._generate_erdos_renyi,
            "barabasi_albert": self._generate_barabasi_albert,
            "ba": self._generate_barabasi_albert,
            "holme_kim": self._generate_holme_kim,
            "hk": self._generate_holme_kim,
            "watts_strogatz": self._generate_watts_strogatz,
            "ws": self._generate_watts_strogatz,
        }
        
        # super args
        super(NodeGeneratorBase, self).__init__(
            only_instance_for_us=only_instance_for_us,
            num_threads=num_threads,
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
        
        # args for generate
        self.nodes_num_min = nodes_num_min
        self.nodes_num_max = nodes_num_max
        self.graph_weighted = graph_weighted
        self.er_prob = er_prob
        self.ba_conn_degree = ba_conn_degree
        self.hk_prob = hk_prob
        self.hk_conn_degree = hk_conn_degree
        self.ws_prob = ws_prob
        self.ws_ring_neighbors = ws_ring_neighbors
        
        # check weighted
        if not only_instance_for_us:
            self._check_weighted()

    def _random_weight(self, n, mu=1, sigma=0.1):
        weights: np.ndarray = np.around(np.random.normal(mu, sigma, n))
        return weights.astype(int).clip(min=0)
    
    def _if_need_weighted(self, nx_graph: nx.Graph):
        if self.graph_weighted:
            weight_mapping = {
                vertex: int(weight)
                for vertex, weight in zip(
                    nx_graph.nodes,
                    self._random_weight(
                        nx_graph.number_of_nodes(), sigma=30, mu=100
                    ),
                )
            }
            nx.set_node_attributes(nx_graph, values=weight_mapping, name="weight")
        return nx_graph

    def _generate_erdos_renyi(self) -> nx.Graph:
        num_nodes = random.randint(self.nodes_num_min, self.nodes_num_max)
        nx_graph = nx.erdos_renyi_graph(num_nodes, self.er_prob)
        return self._if_need_weighted(nx_graph)

    def _generate_barabasi_albert(self) -> nx.Graph:
        num_nodes = random.randint(self.nodes_num_min, self.nodes_num_max)
        nx_graph = nx.barabasi_albert_graph(num_nodes, min(self.ba_conn_degree, num_nodes))
        return self._if_need_weighted(nx_graph)

    def _generate_holme_kim(self) -> nx.Graph:
        num_nodes = random.randint(self.nodes_num_min, self.nodes_num_max)
        nx_graph = nx.powerlaw_cluster_graph(
            num_nodes, min(self.hk_conn_degree, num_nodes), self.hk_prob
        )
        return self._if_need_weighted(nx_graph)

    def _generate_watts_strogatz(self) -> nx.Graph:
        num_nodes = random.randint(self.nodes_num_min, self.nodes_num_max)
        nx_graph = nx.watts_strogatz_graph(num_nodes, self.ws_ring_neighbors, self.ws_prob)
        return self._if_need_weighted(nx_graph)
    
    def _check_weighted(self):
        # check weighted
        if self.graph_weighted != self.solver.weighted:
            message = "``graph_weighted`` and ``solver.weighted`` do not match."
            raise ValueError(message)

        # not support weighted
        if self.graph_weighted == True:
            raise NotImplementedError(
                "The current version does not currently support weighted graphs"
            )     