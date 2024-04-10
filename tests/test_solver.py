import os
import sys

root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from data4co.solver import TSPLKHSolver, TSPConcordeSolver, KaMISSolver
from data4co.utils.mis_utils import cnf_folder_to_gpickle_folder


##############################################
#             Test Func For TSP              #
##############################################


def _test_tsp_lkh_solver():
    tsp_lkh_solver = TSPLKHSolver(lkh_max_trials=100)
    tsp_lkh_solver.from_txt("tests/solver_test/tsp50_test.txt")
    tsp_lkh_solver.solve(show_time=True, num_threads=2)
    _, _, gap_avg, _ = tsp_lkh_solver.evaluate(calculate_gap=True)
    print(f"TSPLKHSolver Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of TSP50 solved by TSPLKHSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)


def _test_tsp_concorde_solver():
    tsp_lkh_solver = TSPConcordeSolver()
    tsp_lkh_solver.from_txt("tests/solver_test/tsp50_test.txt")
    tsp_lkh_solver.solve(show_time=True, num_threads=2)
    _, _, gap_avg, _ = tsp_lkh_solver.evaluate(calculate_gap=True)
    print(f"TSPConcordeSolver Gap: {gap_avg}")
    if gap_avg >= 1e-3:
        message = (
            f"The average gap ({gap_avg}) of TSP50 solved by TSPConcordeSolver "
            "is larger than or equal to 1e-3%."
        )
        raise ValueError(message)


def test_tsp():
    """
    Test TSPSolver
    """
    _test_tsp_lkh_solver()
    _test_tsp_concorde_solver()


##############################################
#            Test Func For KaMIS             #
##############################################


def _test_kamis_solver():
    kamis_solver = KaMISSolver(time_limit=30)
    cnf_folder_to_gpickle_folder(
        cnf_folder="tests/solver_test/solver_test/mis_test",
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
