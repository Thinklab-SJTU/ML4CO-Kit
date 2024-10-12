from ml4co_kit.data.tsp.tsp_uniform import TSPUniformDataset
from ml4co_kit.solver.tsp.base import TSPSolver


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
        **solver_args,
    ):
        solver.from_txt(file_path, ref=True)
        solver.solve(**solver_args)
        return solver.evaluate(calculate_gap=True)
