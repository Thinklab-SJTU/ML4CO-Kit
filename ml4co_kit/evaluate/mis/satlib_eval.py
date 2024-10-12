import os
from ml4co_kit.solver.mis.base import MISSolver
from ml4co_kit.data.mis.satlib_original import SATLIBOriDataset
from ml4co_kit.utils.mis_utils import cnf_folder_to_gpickle_folder


class SATLIBEvaluator:
    def __init__(self, test_folder: str = "dataset/satlib_original/test_files") -> None:
        self.dataset = SATLIBOriDataset()
        self.test_folder = test_folder
        gpickle_root_foler = test_folder + "_gpickle"
        cnf_folder_to_gpickle_folder(
            cnf_folder=test_folder,
            gpickle_foler=gpickle_root_foler
        )
        self.gpickle_foler = os.path.join(gpickle_root_foler, "instance")
        self.ref_solution_path = os.path.join(gpickle_root_foler, "ref_solution.txt")

    def evaluate(self, solver: MISSolver, **solver_args):
        solver.from_gpickle_result_folder(gpickle_folder_path=self.gpickle_foler, ref=False)
        solver.solve(**solver_args)
        solver.from_txt_only_sel_nodes_num(self.ref_solution_path, ref=True)
        return solver.evaluate(calculate_gap=True)