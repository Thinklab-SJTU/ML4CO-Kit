import os
import time
import pickle
import pathlib
import networkx as nx
from tqdm import tqdm
from typing import Union, List
from ml4co_kit.utils.graph.mis import MISGraphData
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.generator.base import NodeGeneratorBase
from ml4co_kit.solver import MISSolver, MISGurobiSolver, KaMISSolver


class MISDataGenerator(NodeGeneratorBase):
    def __init__(
        self,
        only_instance_for_us: bool = False,
        num_threads: int = 1,
        data_type: str = "er",
        solver: Union[SOLVER_TYPE, MISSolver, str] = SOLVER_TYPE.GUROBI,
        train_samples_num: int = 128000,
        val_samples_num: int = 1280,
        test_samples_num: int = 1280,
        save_path: pathlib.Path = "data/mis",
        filename: str = None,
        # args for generate
        nodes_num_min: int = 700,
        nodes_num_max: int = 800,
        graph_weighted: bool = False,
        er_prob: float = 0.5,
        ba_conn_degree: int = 10,
        hk_prob: float = 0.5,
        hk_conn_degree: int = 10,
        ws_prob: float = 0.5,
        ws_ring_neighbors: int = 2,
    ):
        # filename
        if filename is None:
            filename = f"mis_{data_type}-{nodes_num_min}-{nodes_num_max}"
        
        # re-define
        supported_solver_dict = {
            SOLVER_TYPE.GUROBI: MISGurobiSolver,
            SOLVER_TYPE.KAMIS: KaMISSolver
        }
        check_solver_dict = {
            SOLVER_TYPE.GUROBI: self._check_free,
            SOLVER_TYPE.KAMIS: self._check_free
        }

        # super args
        super(MISDataGenerator, self).__init__(
            only_instance_for_us=only_instance_for_us,
            num_threads=num_threads,
            data_type=data_type,
            solver=solver,
            train_samples_num=train_samples_num,
            val_samples_num=val_samples_num,
            test_samples_num=test_samples_num,
            save_path=save_path,
            filename=filename,
            nodes_num_min=nodes_num_min,
            nodes_num_max=nodes_num_max,
            graph_weighted=graph_weighted,
            er_prob=er_prob,
            ba_conn_degree=ba_conn_degree,
            hk_prob=hk_prob,
            hk_conn_degree=hk_conn_degree,
            ws_prob=ws_prob,
            ws_ring_neighbors=ws_ring_neighbors,
            supported_solver_dict=supported_solver_dict,
            check_solver_dict=check_solver_dict
        )
        self.solver: MISSolver
        self._get_filename_for_kamis()
        
    def _get_filename_for_kamis(self):
        if self.solver_type != SOLVER_TYPE.KAMIS:
            return
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        for sample_type in self.sample_types:
            path = os.path.join(self.save_path, sample_type)
            setattr(self, f"{sample_type}_save_path", path)
            if not os.path.exists(os.path.join(path, "instance")):
                os.makedirs(os.path.join(path, "instance"))
            if not os.path.exists(os.path.join(path, "solution")):
                os.makedirs(os.path.join(path, "solution"))

    def generate_only_instance_for_us(self, samples: int) -> List[MISGraphData]:
        nx_graphs = [self.generate_func() for _ in range(samples)]
        self.solver.from_nx_graph(nx_graphs=nx_graphs)
        return self.solver.graph_data
  
    def generate(self):
        if self.solver_type == SOLVER_TYPE.KAMIS:
            # check
            if self.num_threads != 1:
                raise NotImplementedError(
                    "``KaMISSolver`` only supports single-threaded execution"
                )
                
            # generate data
            for sample_type in self.sample_types:
                samples_num = getattr(self, f"{sample_type}_samples_num")
                for idx in tqdm(range(samples_num),desc=self.solver.solve_msg):
                    filename = f"{self.filename}_{idx}"
                    nx_graph: nx.Graph = self.generate_func()
                    output_file = os.path.join(
                        getattr(self, f"{sample_type}_save_path"), "instance", f"{filename}.gpickle"
                    )
                    with open(output_file, "wb") as f:
                        pickle.dump(nx_graph, f, pickle.HIGHEST_PROTOCOL)

            # generate solution
            for sample_type in self.sample_types:
                folder = getattr(self, f"{sample_type}_save_path")
                self.solver.solve(
                    os.path.join(folder, "instance"),
                    os.path.join(folder, "solution")
                )
        else:
            start_time = time.time()
            for _ in tqdm(
                range(self.samples_num // self.num_threads),
                desc=self.solver.solve_msg,
            ):
                # call generate_func to generate the points
                nx_graphs = [self.generate_func() for _ in range(self.num_threads)]
                
                # solve
                self.solver.from_nx_graph(nx_graphs=nx_graphs)
                graph_data = self.solver.solve(num_threads=self.num_threads)
                    
                # write to txt
                with open(self.file_save_path, "a+") as f:
                    for graph in graph_data:
                        graph: MISGraphData
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
                f"Completed generation of {self.samples_num} samples of MIS."
            )
            print(f"Total time: {end_time/60:.1f}m")
            print(f"Average time: {end_time/self.samples_num:.1f}s")
            self.devide_file()            