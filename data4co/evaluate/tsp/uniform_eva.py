from data4co.data.tsp.tsp_uniform import TSPUniformDataset
from data4co.solver.tsp.base import TSPSolver


class TSPUniformEvaluator:
    def __init__(self) -> None:
        self.dataset = TSPUniformDataset()
        self.supported = self.dataset.supported
    
    def show_files(self, nodes_num: int):
        return self.supported[nodes_num]
    
    def evaluate(
        self, 
        solver: TSPSolver,
        file_path: str,
        solver_args: dict={},
    ):
        solver.from_txt(file_path)
        solver.solve(vars(**solver_args))
        return solver.evaluate(caculate_gap=True)
        