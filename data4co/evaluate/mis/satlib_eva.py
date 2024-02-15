import os
from data4co.solver.mis.base import MISSolver
from data4co.data.mis.satlib import SATLIBDataset


class SATLIBEvaluator:
    def __init__(
        self, 
        test_folder: str="dataset/satlib/test_data",
        samples_num: int=-1
    ) -> None:
        self.dataset = SATLIBDataset()
        pickle_path, gpickle_path, _ = self.dataset.generate_mis_from_sat(
            src=test_folder, 
            samples_num=samples_num
        )
        self.pickle_path = pickle_path
        self.test_path = gpickle_path
        self.result_path = os.path.join(self.test_path, "solve")
        
    def evaluate(
        self, 
        solver: MISSolver,
    ):
        solver.from_satlib_pickle(self.pickle_path)
        solver.solve(self.test_path, self.result_path)
        solver.from_folder(self.test_path, self.result_path)
        return solver.evaluate(caculate_gap=True)
        