import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
from ml4co_kit.solver import (
    TSPSolver, TSPLKHSolver, TSPConcordeSolver, TSPGAEAXSolver, 
    TSPGAEAXLargeSolver, KaMISSolver, CVRPSolver, CVRPPyVRPSolver, 
    CVRPLKHSolver, CVRPHGSSolver, ATSPSolver, ATSPLKHSolver
)
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


def _test_tsp_ga_eax_solver(show_time: bool, num_threads: int):
    tsp_ga_eax_solver = TSPGAEAXSolver()
    tsp_ga_eax_solver.from_txt("tests/solver_test/tsp50_test.txt")
    tsp_ga_eax_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = tsp_ga_eax_solver.evaluate(calculate_gap=True)
    print(f"TSPGAEAXSolver Gap: {gap_avg}")
    if gap_avg >= 1e-3:
        message = (
            f"The average gap ({gap_avg}) of TSP50 solved by TSPGAEAXSolver "
            "is larger than or equal to 1e-3%."
        )
        raise ValueError(message)
    

def test_tsp_ga_eax_solver():
    _test_tsp_ga_eax_solver(True, 1)
    _test_tsp_ga_eax_solver(True, 2)
    _test_tsp_ga_eax_solver(False, 1)
    _test_tsp_ga_eax_solver(False, 2)


def _test_tsp_ga_eax_large_solver(show_time: bool, num_threads: int):
    tsp_ga_eax_large_solver = TSPGAEAXLargeSolver()
    tsp_ga_eax_large_solver.from_txt("tests/solver_test/tsp1000_test.txt")
    tsp_ga_eax_large_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = tsp_ga_eax_large_solver.evaluate(calculate_gap=True)
    print(f"TSPGAEAXLargeSolver Gap: {gap_avg}")
    if gap_avg >= 5e-2:
        message = (
            f"The average gap ({gap_avg}) of TSP1000 solved by TSPGAEAXLargeSolver "
            "is larger than or equal to 5e-2%."
        )
        raise ValueError(message)


def test_tsp_ga_eax_large_solver():
    _test_tsp_ga_eax_large_solver(True, 1)
    _test_tsp_ga_eax_large_solver(True, 2)
    _test_tsp_ga_eax_large_solver(False, 1)
    _test_tsp_ga_eax_large_solver(False, 2)


def test_tsp():
    """
    Test TSPSolver
    """
    test_tsp_base_solver()
    test_tsp_lkh_solver()
    test_tsp_concorde_solver()
    test_tsp_ga_eax_solver()
    test_tsp_ga_eax_large_solver()


##############################################
#             Test Func For MIS              #
##############################################


def test_mis_kamis_solver():
    kamis_solver = KaMISSolver(time_limit=10)
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
    test_mis_kamis_solver()


##############################################
#             Test Func For CVRP             #
##############################################

def test_cvrp_base_solver():
    solver = CVRPSolver()
    solver.from_txt("tests/solver_test/cvrp50_test.txt")
    os.remove("tests/solver_test/cvrp50_test.txt")
    solver.read_tours(solver.ref_tours)
    solver.to_vrp(
        save_dir="tests/solver_test/cvrp50_test",
        filename="problem",
        dtype="float"
    )
    solver.to_sol(
        save_dir="tests/solver_test/cvrp50_test",
        filename="solution",
        dtype="float"
    )
    solver.to_txt("tests/solver_test/cvrp50_test.txt")


def _test_cvrp_pyvrp_solver(show_time: bool, num_threads: int):
    cvrp_pyvrp_solver = CVRPPyVRPSolver(time_limit=10)
    cvrp_pyvrp_solver.from_txt("tests/solver_test/cvrp50_test_small.txt")
    cvrp_pyvrp_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = cvrp_pyvrp_solver.evaluate(calculate_gap=True)
    print(f"CVRPPyVRPSolver Gap: {gap_avg}")
    if gap_avg >= 1e-3:
        message = (
            f"The average gap ({gap_avg}) of CVRP50 solved by CVRPPyVRPSolver "
            "is larger than or equal to 1e-3%."
        )
        raise ValueError(message)


def test_cvrp_pyvrp_solver():
    _test_cvrp_pyvrp_solver(True, 1)
    _test_cvrp_pyvrp_solver(True, 2)
    _test_cvrp_pyvrp_solver(False, 1)
    _test_cvrp_pyvrp_solver(False, 2)


def _test_cvrp_lkh_solver(show_time: bool, num_threads: int):
    cvrp_lkh_solver = CVRPLKHSolver(lkh_max_trials=500, lkh_runs=10)
    cvrp_lkh_solver.from_txt("tests/solver_test/cvrp50_test_small.txt")
    cvrp_lkh_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = cvrp_lkh_solver.evaluate(calculate_gap=True)
    print(f"CVRPLKHSolver Gap: {gap_avg}")
    if gap_avg >= 1e-3:
        message = (
            f"The average gap ({gap_avg}) of CVRP50 solved by CVRPLKHSolver "
            "is larger than or equal to 1e-3%."
        )
        raise ValueError(message)


def test_cvrp_lkh_solver():
    _test_cvrp_lkh_solver(True, 1)
    _test_cvrp_lkh_solver(True, 2)
    _test_cvrp_lkh_solver(False, 1)
    _test_cvrp_lkh_solver(False, 2)


def _test_cvrp_hgs_solver(show_time: bool, num_threads: int):
    cvrp_hgs_solver = CVRPHGSSolver(time_limit=0.5)
    cvrp_hgs_solver.from_txt("tests/solver_test/cvrp50_test_small.txt")
    cvrp_hgs_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = cvrp_hgs_solver.evaluate(calculate_gap=True)
    print(f"CVRPHGSSolver Gap: {gap_avg}")
    if gap_avg >= 1e-4:
        message = (
            f"The average gap ({gap_avg}) of CVRP50 solved by CVRPHGSSolver "
            "is larger than or equal to 1e-4%."
        )
        raise ValueError(message)


def test_cvrp_hgs_solver():
    _test_cvrp_hgs_solver(True, 1)
    _test_cvrp_hgs_solver(True, 2)
    _test_cvrp_hgs_solver(False, 1)
    _test_cvrp_hgs_solver(False, 2)


def test_cvrp():
    """
    Test CVRPSolver
    """
    test_cvrp_base_solver()
    test_cvrp_pyvrp_solver()
    test_cvrp_lkh_solver()
    test_cvrp_hgs_solver()
    

##############################################
#             Test Func For TSP              #
##############################################

def test_atsp_base_solver():
    solver = ATSPSolver()
    solver.from_txt("tests/solver_test/atsp55_test.txt")
    solver.read_tours(solver.ref_tours)
    solver.to_atsp_folder("tests/solver_test/atsp_test/problem", filename="atsp55")
    solver.from_atsp_folder("tests/solver_test/atsp_test/problem")
    solver.to_opt_tour_folder("tests/solver_test/atsp_test/solution", filename="astsp55")
    solver.read_ref_tours_from_folder("tests/solver_test/atsp_test/solution")


def _test_atsp_lkh_solver(show_time: bool, num_threads: int):
    tsp_lkh_solver = ATSPLKHSolver(scale=1000, lkh_max_trials=500)
    tsp_lkh_solver.from_txt("tests/solver_test/atsp55_test.txt")
    tsp_lkh_solver.solve(show_time=show_time, num_threads=num_threads)
    costs_avg = tsp_lkh_solver.evaluate(calculate_gap=False)
    print(f"ATSPLKHSolver Cost Avg: {costs_avg}")
    if costs_avg >= 0.1:
        message = (
            f"The average cost ({costs_avg}) of ATSP50 solved by ATSPLKHSolver "
            "is larger than or equal to 0.1."
        )
        raise ValueError(message)


def test_atsp_lkh_solver():
    _test_atsp_lkh_solver(True, 1)
    _test_atsp_lkh_solver(True, 2)
    _test_atsp_lkh_solver(False, 1)
    _test_atsp_lkh_solver(False, 2)
    
    
def test_atsp():
    """
    Test ATSPSolver
    """
    test_atsp_base_solver()
    test_atsp_lkh_solver()

    
##############################################
#                    MAIN                    #
##############################################

if __name__ == "__main__":
    # test_tsp()
    # test_mis()
    # test_cvrp()
    test_atsp()
