import os
import time
import pathlib
import random
import numpy as np
import networkx as nx
from tqdm import tqdm
from typing import Union, List
from ml4co_kit.utils.graph.mcut import MCutGraphData
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.solver import MCutSolver, MCutGurobiSolver


class MCutDataGenerator:
    def __init__(
        self,
        only_instance_for_us: bool = False,
        num_threads: int = 1,
        nodes_num_min: int = 700,
        nodes_num_max: int = 800,
        data_type: str = "er",
        solver: Union[SOLVER_TYPE, MCutSolver] = SOLVER_TYPE.GUROBI,
        train_samples_num: int = 128000,
        val_samples_num: int = 1280,
        test_samples_num: int = 1280,
        save_path: pathlib.Path = "data/mcut/er",
        filename: str = None,
        # args for generate
        graph_weighted: bool = False,
        er_prob: float = 0.5,
        ba_conn_degree: int = 10,
        hk_prob: float = 0.5,
        hk_conn_degree: int = 10,
        ws_prob: float = 0.5,
        ws_ring_neighbors: int = 2,
    ):
        """
        MCutDataGenerator
        Args:
            nodes_num_min (int, optional):
                The minimum number of nodes.
            nodes_num_max (int, optional):
                The maximum number of nodes.
            data_type (str, optional):
                The data type. Support: ``erdos_renyi``,``er``,``barabasi_albert``,
                ``ba``,``holme_kim``,``hk``,``watts_strogatz``,``ws``.
            solver_type (str, optional):
                The solver type. Support: ``gurobi``.
            train_samples_num (int, optional):
                The number of training samples.
            val_samples_num (int, optional):
                The number of validation samples.
            test_samples_num (int, optional):
                The number of test samples.
            save_path (pathlib.Path, optional):
                The save path of mcut samples/datasets.
            filename (str, optional):
                The filename of mcut samples.
            graph_weighted (bool, optional):
                If enabled, generate the weighted MCut problem instead of MCut.
            er_prob (float, optional):
                The probability parameter for Erdos-Renyi graph generation.
            ba_conn_degree (int, optional):
                The connection degree parameter for Barabasi-Albert graph generation.
            hk_prob (float, optional):
                The probability parameter for Hyperbolic graph generation.
            hk_conn_degree (int, optional):
                The connection degree parameter for Hyperbolic graph generation.
            ws_prob (float, optional):
                The probability parameter for Watts-Strogatz graph generation.
            ws_ring_neighbors (int, optional):
                The number of ring neighbors for Watts-Strogatz graph generation.
        """
        # record variable data
        self.num_threads = num_threads
        self.nodes_num_min = nodes_num_min
        self.nodes_num_max = nodes_num_max
        self.data_type = data_type
        self.solver = solver
        self.train_samples_num = train_samples_num
        self.val_samples_num = val_samples_num
        self.test_samples_num = test_samples_num
        self.save_path = save_path
        self.filename = filename
        
        # args for generate
        self.graph_weighted = graph_weighted
        self.er_prob = er_prob
        self.ba_conn_degree = ba_conn_degree
        self.hk_prob = hk_prob
        self.hk_conn_degree = hk_conn_degree
        self.ws_prob = ws_prob
        self.ws_ring_neighbors = ws_ring_neighbors

        # only instance for us
        self.only_instance_for_us = only_instance_for_us
        self.check_data_type()
        
        # generate and solve
        if only_instance_for_us == False:
            # check the input variables
            self.sample_types = ["train", "val", "test"]
            self.check_num_threads()    
            self.check_solver()
            self.check_save_path()
            self.get_filename()

    def check_num_threads(self):
        self.samples_num = 0
        for sample_type in self.sample_types:
            self.samples_num += getattr(self, f"{sample_type}_samples_num")
            if self.samples_num % self.num_threads != 0:
                message = "``samples_num`` must be divisible by the number of threads. "
                raise ValueError(message)

    def check_num_threads(self):
        self.samples_num = 0
        for sample_type in self.sample_types:
            self.samples_num += getattr(self, f"{sample_type}_samples_num")
            if self.samples_num % self.num_threads != 0:
                message = "``samples_num`` must be divisible by the number of threads. "
                raise ValueError(message)

    def check_data_type(self):
        generate_func_dict = {
            "erdos_renyi": self.generate_erdos_renyi,
            "er": self.generate_erdos_renyi,
            "barabasi_albert": self.generate_barabasi_albert,
            "ba": self.generate_barabasi_albert,
            "holme_kim": self.generate_holme_kim,
            "hk": self.generate_holme_kim,
            "watts_strogatz": self.generate_watts_strogatz,
            "ws": self.generate_watts_strogatz,
        }
        supported_data_type = generate_func_dict.keys()
        if self.data_type not in supported_data_type:
            message = (
                f"The input data type ({self.data_type}) is not a valid type, "
                f"and the generator only supports {supported_data_type}."
            )
            raise ValueError(message)
        self.generate_func = generate_func_dict[self.data_type]

    def check_save_path(self):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        for sample_type in self.sample_types:
            path = os.path.join(self.save_path, sample_type)
            setattr(self, f"{sample_type}_save_path", path)

    def get_filename(self):
        if self.filename is None:
            self.filename = (
                f"mcut_{self.data_type}_{self.nodes_num_min}_{self.nodes_num_max}"
            )
        self.file_save_path = os.path.join(self.save_path, self.filename + ".txt")
        for sample_type in self.sample_types:
            setattr(
                self,
                f"{sample_type}_file_save_path",
                os.path.join(self.save_path, self.filename + f"_{sample_type}.txt"),
            )
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def check_solver(self):
        # check solver
        if isinstance(self.solver, SOLVER_TYPE):
            self.solver_type = self.solver
            supported_solver_dict = {
                SOLVER_TYPE.GUROBI: MCutGurobiSolver
            }
            supported_solver_type = supported_solver_dict.keys()
            if self.solver not in supported_solver_type:
                message = (
                    f"The input solver type ({self.solver}) is not a valid type, "
                    f"and the generator only supports {supported_solver_type}."
                )
                raise ValueError(message)
            self.solver = supported_solver_dict[self.solver]()
        else:
            self.solver: MCutSolver
            self.solver_type = self.solver.solver_type
        
        # check solver
        check_solver_dict = {
            SOLVER_TYPE.GUROBI: self.check_free,
        }
        check_func = check_solver_dict[self.solver_type]
        check_func()
          
        # check weighted
        if self.graph_weighted != self.solver.weighted:
            message = "``graph_weighted`` and ``solver.weighted`` do not match."
            raise ValueError(message)
        
    def check_free(self):
        return
    
    def random_weight(self, n, mu=1, sigma=0.1):
        return np.around(np.random.normal(mu, sigma, n)).astype(int).clip(min=0)

    def generate_only_instance_for_us(self, samples: int) -> List[MCutGraphData]:
        nx_graphs = [self.generate_func() for _ in range(samples)]
        self.solver.from_nx_graph(nx_graphs=nx_graphs)
        return self.solver.graph_data
    
    def generate(self):
        start_time = time.time()
        for _ in tqdm(
            range(self.samples_num // self.num_threads),
            desc=f"Solving MCut Using {self.solver_type}",
        ):
            # call generate_func to generate the points
            nx_graphs = [self.generate_func() for _ in range(self.num_threads)]
            
            # solve
            self.solver.from_nx_graph(nx_graphs=nx_graphs)
            graph_data = self.solver.solve(num_threads=self.num_threads)
            
            # write to txt
            with open(self.file_save_path, "a+") as f:
                for graph in graph_data:
                    graph: MCutGraphData
                    edge_index = graph.edge_index.T
                    nodes_label = graph.nodes_label
                    f.write(" ".join(str(src) + str(" ") + str(tgt) for src, tgt in edge_index))
                    f.write(str(" ") + str("label") + str(" "))
                    f.write(str(" ").join(str(node_label) for node_label in nodes_label))
                    f.write("\n")
            f.close()
        
        # info
        end_time = time.time() - start_time
        print(
            f"Completed generation of {self.samples_num} samples of MCut."
        )
        print(f"Total time: {end_time/60:.1f}m")
        print(f"Average time: {end_time/self.samples_num:.1f}s")
        self.devide_file()            

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

    def if_need_weighted(self, nx_graph: nx.Graph):
        if self.graph_weighted:
            weight_mapping = {
                vertex: int(weight)
                for vertex, weight in zip(
                    nx_graph.nodes,
                    self.random_weight(
                        nx_graph.number_of_nodes(), sigma=30, mu=100
                    ),
                )
            }
            nx.set_node_attributes(nx_graph, values=weight_mapping, name="weight")
        return nx_graph
    
    def generate_erdos_renyi(self) -> nx.Graph:
        num_nodes = random.randint(self.nodes_num_min, self.nodes_num_max)
        nx_graph = nx.erdos_renyi_graph(num_nodes, self.er_prob)
        return self.if_need_weighted(nx_graph)

    def generate_barabasi_albert(self) -> nx.Graph:
        num_nodes = random.randint(self.nodes_num_min, self.nodes_num_max)
        nx_graph = nx.barabasi_albert_graph(num_nodes, min(self.ba_conn_degree, num_nodes))
        return self.if_need_weighted(nx_graph)

    def generate_holme_kim(self) -> nx.Graph:
        num_nodes = random.randint(self.nodes_num_min, self.nodes_num_max)
        nx_graph = nx.powerlaw_cluster_graph(
            num_nodes, min(self.hk_conn_degree, num_nodes), self.hk_prob
        )
        return self.if_need_weighted(nx_graph)

    def generate_watts_strogatz(self) -> nx.Graph:
        num_nodes = random.randint(self.nodes_num_min, self.nodes_num_max)
        nx_graph = nx.watts_strogatz_graph(num_nodes, self.ws_ring_neighbors, self.ws_prob)
        return self.if_need_weighted(nx_graph)