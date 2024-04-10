import os
import sys

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from data4co.data import TSPLIBOriDataset, TSPUniformDataset, SATLIBOriDataset
from data4co.evaluate import TSPLIBOriEvaluator, TSPUniformEvaluator
from data4co.solver import TSPConcordeSolver


def test_tsp_dataset():
    TSPLIBOriDataset()
    TSPUniformDataset()


def test_tsplib_original_eval():
    eva = TSPLIBOriEvaluator()
    con_solver = TSPConcordeSolver(scale=1)
    result = eva.evaluate(con_solver, norm="EUC_2D")
    gap_avg = result["gaps"][-1]
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of TSPLIB(EUC_2D) solved by TSPConcordeSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)
    eva.evaluate(con_solver, norm="GEO")
    gap_avg = result["gaps"][-1]
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of TSPLIB(GEO) solved by TSPConcordeSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)


def test_tsp_uniform_eval():
    eva = TSPUniformEvaluator()
    supported_files = eva.show_files(nodes_num=50)
    test_file_path = supported_files[-1]
    con_solver = TSPConcordeSolver()
    _, _, gap_avg, _ = eva.evaluate(
        solver=con_solver, 
        file_path=test_file_path,
        num_threads=2,
        show_time=True
    )
    print(f"TSPConcordeSolver Gap: {gap_avg}")
    if gap_avg >= 1e-3:
        message = (
            f"The average gap ({gap_avg}) of TSP50 solved by TSPConcordeSolver "
            "is larger than or equal to 1e-3%."
        )
        raise ValueError(message)


def test_sat_dataset():
    SATLIBOriDataset()


if __name__ == "__main__":
    test_tsp_dataset()
    test_sat_dataset()
    test_tsplib_original_eval()
    test_tsp_uniform_eval()
