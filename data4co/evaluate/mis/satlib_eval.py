import os
from data4co.solver.mis.base import MISSolver
from data4co.data.mis.satlib_original import SATLIBOriDataset
from data4co.utils.mis_utils import cnf_folder_to_gpickle_folder


class SATLIBEvaluator:
    def __init__(
        self, test_folder: str = "dataset/satlib_original/test_files", samples_num: int = -1
    ) -> None:
        self.dataset = SATLIBOriDataset()
        self.test_folder = test_folder
        gpickle_root_foler = test_folder + "_gpickle"
        cnf_folder_to_gpickle_folder(
            cnf_folder=test_folder,
            gpickle_foler=gpickle_root_foler
        )
        self.gpickle_foler = os.path.join(gpickle_root_foler, "mis_graph")
        self.ref_solution_path = os.path.join(gpickle_root_foler, "ref_solution.txt")

    def evaluate(self, solver: MISSolver, **solver_args):
        solver.solve(self.gpickle_foler, **solver_args)
        solver.from_gpickle_folder(self.gpickle_foler, ref=False)
        solver.read_ref_sel_nodes_num_from_txt(self.ref_solution_path)
        return solver.evaluate(calculate_gap=True)
