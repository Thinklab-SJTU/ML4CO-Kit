from data4co.data.tsp.tspconcorde import TSPConcordeDataset
from data4co.solver.tsp.base import TSPSolver


class TSPUniformEvaluator:
    def __init__(self, num_nodes: int) -> None:
        self.dataset = TSPConcordeDataset()
        self.data_path = self.dataset.get_data_path(num_nodes)
    
    def evaluate(
        self, 
        solver: TSPSolver,
        **kwargs
    ):
        solver.from_txt(self.data_path)
        batch_size = kwargs.get("batch_size", 1)
        solver.solve(batch_size=batch_size)
        return solver.evaluate(caculate_gap=True)
        