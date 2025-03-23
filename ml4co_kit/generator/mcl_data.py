import pathlib
from typing import Union, List
from ml4co_kit.utils.graph.mcl import MClGraphData
from ml4co_kit.utils.type_utils import SOLVER_TYPE
from ml4co_kit.generator.base import NodeGeneratorBase
from ml4co_kit.solver import MClSolver, MClGurobiSolver


class MClDataGenerator(NodeGeneratorBase):
    def __init__(
        self,
        only_instance_for_us: bool = False,
        num_threads: int = 1,
        data_type: str = "er",
        solver: Union[SOLVER_TYPE, MClSolver, str] = SOLVER_TYPE.GUROBI,
        train_samples_num: int = 128000,
        val_samples_num: int = 1280,
        test_samples_num: int = 1280,
        save_path: pathlib.Path = "data/mcl",
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
        rb_n_scale: tuple = (20, 25),
        rb_k_scale: tuple = (5, 12),
        rb_p_scale: tuple = (0.3, 1.0)
    ):
        # filename
        if filename is None:
            filename = f"mcl_{data_type}-{nodes_num_min}-{nodes_num_max}"
        
        # re-define
        supported_solver_dict = {
            SOLVER_TYPE.GUROBI: MClGurobiSolver
        }
        check_solver_dict = {
            SOLVER_TYPE.GUROBI: self._check_free
        }
        
        # super args
        super(MClDataGenerator, self).__init__(
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
            rb_n_scale=rb_n_scale,
            rb_k_scale=rb_k_scale,
            rb_p_scale=rb_p_scale,
            supported_solver_dict=supported_solver_dict,
            check_solver_dict=check_solver_dict
        )
        self.solver: MClSolver
        
    def generate_only_instance_for_us(self, samples: int) -> List[MClGraphData]:
        nx_graphs = [self.generate_func() for _ in range(samples)]
        self.solver.from_nx_graph(nx_graphs=nx_graphs)
        return self.solver.graph_data
    
    def _generate_core(self):
        # call generate_func to generate the points
        nx_graphs = [self.generate_func() for _ in range(self.num_threads)]

        # solve
        self.solver.from_nx_graph(nx_graphs=nx_graphs)
        graph_data = self.solver.solve(num_threads=self.num_threads)
        
        # write to txt
        with open(self.file_save_path, "a+") as f:
            for graph in graph_data:
                graph: MClGraphData
                edge_index = graph.edge_index.T
                nodes_label = graph.nodes_label
                f.write(" ".join(str(src) + str(" ") + str(tgt) for src, tgt in edge_index))
                f.write(str(" ") + str("label") + str(" "))
                f.write(str(" ").join(str(node_label) for node_label in nodes_label))
                f.write("\n")
        f.close()