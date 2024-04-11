import os
import sys

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4co_kit.solver import TSPSolver, TSPLKHSolver, TSPConcordeSolver, KaMISSolver
from ml4co_kit.utils.mis_utils import cnf_folder_to_gpickle_folder


##############################################
#             Test Func For TSP              #
##############################################

def test_tsp_base_solver():
    solver = TSPSolver()
    solver.from_txt("tests/solver_test/tsp50_test_small.txt")
    os.remove("tests/solver_test/tsp50_test_small.txt")
    solver.read_tours(solver.ref_tours)
    solver.to_tsp(
        save_dir="tests/solver_test/tsp50_test_small",
        filename="problem"
    )
    solver.to_opt_tour(
        save_dir="tests/solver_test/tsp50_test_small",
        filename="solution"
    )
    solver.to_txt("tests/solver_test/tsp50_test_small.txt")
    

def _test_tsp_lkh_solver(show_time: bool, num_threads: int):
    tsp_lkh_solver = TSPLKHSolver(lkh_max_trials=100)
    tsp_lkh_solver.from_txt("tests/solver_test/tsp50_test.txt")
    tsp_lkh_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = tsp_lkh_solver.evaluate(calculate_gap=True)
    print(f"TSPLKHSolver Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of TSP50 solved by TSPLKHSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)


def test_tsp_lkh_solver():
    _test_tsp_lkh_solver(True, 1)
    _test_tsp_lkh_solver(True, 2)
    _test_tsp_lkh_solver(False, 1)
    _test_tsp_lkh_solver(False, 2)
    

def _test_tsp_concorde_solver(show_time: bool, num_threads: int):
    tsp_lkh_solver = TSPConcordeSolver()
    tsp_lkh_solver.from_txt("tests/solver_test/tsp50_test.txt")
    tsp_lkh_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = tsp_lkh_solver.evaluate(calculate_gap=True)
    print(f"TSPConcordeSolver Gap: {gap_avg}")
    if gap_avg >= 1e-3:
        message = (
            f"The average gap ({gap_avg}) of TSP50 solved by TSPConcordeSolver "
            "is larger than or equal to 1e-3%."
        )
        raise ValueError(message)


def test_tsp_concorde_solver():
    _test_tsp_concorde_solver(True, 1)
    _test_tsp_concorde_solver(True, 2)
    _test_tsp_concorde_solver(False, 1)
    _test_tsp_concorde_solver(False, 2)


def test_tsp():
    """
    Test TSPSolver
    """
    test_tsp_base_solver()
    test_tsp_lkh_solver()
    test_tsp_concorde_solver()


##############################################
#            Test Func For KaMIS             #
##############################################


def _test_kamis_solver():
    kamis_solver = KaMISSolver(time_limit=30)
    cnf_folder_to_gpickle_folder(
        cnf_folder="tests/solver_test/mis_test_cnf",
        gpickle_foler="tests/solver_test/mis_test"
    )
    kamis_solver.solve(
        src="tests/solver_test/mis_test/mis_graph", 
        out="tests/solver_test/mis_test/mis_graph/solve"
    )
    kamis_solver.from_gpickle_folder("tests/solver_test/mis_test/mis_graph")
    kamis_solver.read_ref_sel_nodes_num_from_txt("tests/solver_test/mis_test/ref_solution.txt")
    gap_avg = kamis_solver.evaluate(calculate_gap=True)["avg_gap"]
    print(f"KaMISSolver Gap: {gap_avg}")
    if gap_avg >= 0.1:
        message = (
            f"The average gap ({gap_avg}) of MIS solved by KaMISSolver "
            "is larger than or equal to 0.1%."
        )
        raise ValueError(message)


def test_mis():
    """
    Test MISSolver
    """
    _test_kamis_solver()


##############################################
#                    MAIN                    #
##############################################

if __name__ == "__main__":
    test_tsp()
    test_mis()
