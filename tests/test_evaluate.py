import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
import shutil
from ml4co_kit import *


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


def test_tsplib4ml_eval():
    eva = TSPLIB4MLEvaluator()
    lkh_solver = TSPLKHSolver(lkh_max_trials=100)
    result = eva.evaluate(
        lkh_solver, normalize=True,
        min_nodes_num=50, max_nodes_num=200
    )
    gap_avg = result["gaps"][-1]
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of TSPLIB4ML(normalize=True) solved by TSPLKHSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)


def test_tsp_uniform_eval():
    eva = TSPUniformEvaluator()
    supported_files = eva.show_files(nodes_num=50)
    test_file_path = supported_files[-1]
    lkh_solver = TSPLKHSolver(lkh_max_trials=100)
    _, _, gap_avg, _ = eva.evaluate(
        solver=lkh_solver, 
        file_path=test_file_path,
        num_threads=2,
        show_time=True
    )
    print(f"TSPConcordeSolver Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of TSP50 solved by TSPLKHSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)


def test_satlib_original_eval():
    SATLIBOriDataset()
    test_folder_full = "dataset/satlib_original/test_files"
    test_folder_part = "dataset/satlib_original/test_files_part"
    if not os.path.exists(test_folder_part):
        os.mkdir(test_folder_part)
    sel_files = os.listdir(test_folder_full)
    sel_files.sort()
    sel_files = sel_files[:4]
    for filename in sel_files:
        src = os.path.join(test_folder_full, filename)
        dst = os.path.join(test_folder_part, filename)
        shutil.copy(src=src, dst=dst)
    eva = SATLIBEvaluator(test_folder="dataset/satlib_original/test_files_part")
    solver = KaMISSolver(time_limit=1)
    gap_avg = eva.evaluate(
        solver, 
        src="dataset/satlib_original/test_files_part_gpickle/instance", 
        out="dataset/satlib_original/test_files_part_gpickle/solution"
    )[2]
    print(f"KaMISSolver Gap: {gap_avg}")
    if gap_avg >= 0.1:
        message = (
            f"The average gap ({gap_avg}) of MIS solved by KaMISSolver "
            "is larger than or equal to 0.1%."
        )
        raise ValueError(message)


def test_vrplib_original_eval():
    VRPLIBOriDataset()
    

def test_cvrp_uniform_eval():
    CVRPUniformEvaluator()


if __name__ == "__main__":
    test_tsplib_original_eval()
    test_tsplib4ml_eval()
    test_tsp_uniform_eval()
    test_satlib_original_eval()
    test_vrplib_original_eval()
    test_cvrp_uniform_eval()
