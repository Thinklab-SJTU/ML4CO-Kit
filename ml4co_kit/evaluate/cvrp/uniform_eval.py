from ml4co_kit.data.cvrp.cvrp_uniform import CVRPUniformDataset
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
        show_time: bool = False,
        **solver_args,
    ):
        solver.from_txt(file_path, ref=True)
        solver.solve(show_time=show_time, **solver_args)
        return solver.evaluate(calculate_gap=True)