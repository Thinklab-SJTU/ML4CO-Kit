from ml4co_kit.data.vrp.cvrp_uniform import CVRPUniformDataset
from ml4co_kit.solver.cvrp.base import CVRPSolver


class CVRPUniformEvaluator:
    def __init__(self) -> None:
        self.dataset = CVRPUniformDataset()
        self.supported = self.dataset.supported

    def show_files(self, nodes_num: int):
        return self.supported[nodes_num]

    def evaluate(
        self,
        solver: CVRPSolver,
        file_path: str,
        **solver_args,
    ):
        solver.from_txt(file_path)
        solver.solve(**solver_args)
        return solver.evaluate(calculate_gap=True)