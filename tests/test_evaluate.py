import os
import sys
import shutil

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4co_kit.data import SATLIBOriDataset
from ml4co_kit.evaluate import TSPLIBOriEvaluator, TSPUniformEvaluator, SATLIBEvaluator
from ml4co_kit.solver import TSPConcordeSolver, KaMISSolver


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


def test_satlib_original_eval():
    SATLIBOriDataset()
    test_folder_full = "dataset/satlib_original/test_files"
    test_folder_part = "dataset/satlib_original/test_files_part"
    os.mkdir(test_folder_part)
    sel_files = os.listdir(test_folder_full)[:10]
    for filename in sel_files:
        src = os.path.join(test_folder_full, filename)
        dst = os.path.join(test_folder_part, filename)
        shutil.copy(src=src, dst=dst)
    eva = SATLIBEvaluator(test_folder="dataset/satlib_original/test_files_part")
    solver = KaMISSolver(time_limit=10)
    gap_avg = eva.evaluate(solver)["avg_gap"]
    print(f"KaMISSolver Gap: {gap_avg}")
    if gap_avg >= 0.1:
        message = (
            f"The average gap ({gap_avg}) of MIS solved by KaMISSolver "
            "is larger than or equal to 0.1%."
        )
        raise ValueError(message)


if __name__ == "__main__":
    test_tsplib_original_eval()
    test_tsp_uniform_eval()
    test_satlib_original_eval()
