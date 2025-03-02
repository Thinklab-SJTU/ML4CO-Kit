import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_folder)
import shutil
from ml4co_kit import *

GUROBI_TEST = False


##############################################
#             Test Func For ATSP             #
##############################################

def test_atsp_base_solver():
    solver = ATSPSolver()
    solver.from_txt("tests/data_for_tests/solver/atsp/atsp50.txt")
    os.remove("tests/data_for_tests/solver/atsp/atsp50.txt")
    solver.to_tsplib_folder(
        atsp_save_dir="tests/data_for_tests/solver/atsp/atsp50_tsplib_instance",
        atsp_filename="atsp50",
        tour_save_dir="tests/data_for_tests/solver/atsp/atsp50_tsplib_solution",
        tour_filename="atsp50",
        apply_scale=True,
        to_int=True,
        show_time=True
    )
    shutil.rmtree("tests/data_for_tests/solver/atsp/atsp50_tsplib_instance")
    shutil.rmtree("tests/data_for_tests/solver/atsp/atsp50_tsplib_solution")
    solver.to_txt(
        file_path="tests/data_for_tests/solver/atsp/atsp50.txt",
        apply_scale=False,
        to_int=False,
    )
    
    
def _test_atsp_lkh_solver(show_time: bool, num_threads: int):
    atsp_lkh_solver = ATSPLKHSolver(lkh_max_trials=500)
    atsp_lkh_solver.from_txt("tests/data_for_tests/solver/atsp/atsp50.txt", ref=True)
    atsp_lkh_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = atsp_lkh_solver.evaluate(calculate_gap=True)
    print(f"ATSPLKHSolver Gap: {gap_avg}")
    if gap_avg >= 1e-4:
        message = (
            f"The average gap ({gap_avg}) of ATSP50 solved by ATSPLKHSolver "
            "is larger than or equal to 1e-4%."
        )
        raise ValueError(message)


def test_atsp_lkh_solver():
    _test_atsp_lkh_solver(True, 1)
    _test_atsp_lkh_solver(False, 2)
    
    
def test_atsp():
    """
    Test ATSPSolver
    """
    test_atsp_base_solver()
    test_atsp_lkh_solver()


##############################################
#             Test Func For CVRP             #
##############################################

def test_cvrp_base_solver():
    solver = CVRPSolver()
    solver.from_txt("tests/data_for_tests/solver/cvrp/cvrp50.txt", ref=False)
    os.remove("tests/data_for_tests/solver/cvrp/cvrp50.txt")
    solver.to_vrplib_folder(
        vrp_save_dir="tests/data_for_tests/solver/cvrp/cvrp50_vrplib_instance",
        vrp_filename="cvrp50",
        sol_save_dir="tests/data_for_tests/solver/cvrp/cvrp50_vrplib_solution",
        sol_filename="cvrp50",
        apply_scale=True,
        to_int=True,
    )
    shutil.rmtree("tests/data_for_tests/solver/cvrp/cvrp50_vrplib_instance")
    shutil.rmtree("tests/data_for_tests/solver/cvrp/cvrp50_vrplib_solution")
    solver.to_txt("tests/data_for_tests/solver/cvrp/cvrp50.txt")


def _test_cvrp_hgs_solver(show_time: bool, num_threads: int):
    cvrp_hgs_solver = CVRPHGSSolver(time_limit=0.5)
    cvrp_hgs_solver.from_txt("tests/data_for_tests/solver/cvrp/cvrp50.txt", ref=True)
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
    _test_cvrp_hgs_solver(False, 2)


def _test_cvrp_lkh_solver(show_time: bool, num_threads: int):
    cvrp_lkh_solver = CVRPLKHSolver(lkh_max_trials=500, lkh_runs=10)
    cvrp_lkh_solver.from_txt("tests/data_for_tests/solver/cvrp/cvrp50.txt", ref=True)
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
    _test_cvrp_lkh_solver(False, 2)


def _test_cvrp_pyvrp_solver(show_time: bool, num_threads: int):
    cvrp_pyvrp_solver = CVRPPyVRPSolver(time_limit=5)
    cvrp_pyvrp_solver.from_txt("tests/data_for_tests/solver/cvrp/cvrp50.txt", ref=True)
    cvrp_pyvrp_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = cvrp_pyvrp_solver.evaluate(calculate_gap=True)
    print(f"CVRPPyVRPSolver Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of CVRP50 solved by CVRPPyVRPSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)


def test_cvrp_pyvrp_solver():
    _test_cvrp_pyvrp_solver(True, 1)
    _test_cvrp_pyvrp_solver(False, 2)


def test_cvrp():
    """
    Test CVRPSolver
    """
    test_cvrp_base_solver()
    test_cvrp_hgs_solver()
    test_cvrp_lkh_solver()
    test_cvrp_pyvrp_solver()


##############################################
#              Test Func For LP              #
##############################################


def test_lp_base_solver():
    solver = LPSolver()
    solver.from_txt("tests/data_for_tests/solver/lp/lp_20_16.txt", ref=False)
    os.remove("tests/data_for_tests/solver/lp/lp_20_16.txt")
    solver.to_txt("tests/data_for_tests/solver/lp/lp_20_16.txt")
    

def _test_lp_gurobi_solver(show_time: bool, num_threads: int):
    if not GUROBI_TEST:
        return
    gurobi_solver = LPGurobiSolver(time_limit=10.0)
    gurobi_solver.from_txt(
        file_path="tests/data_for_tests/solver/lp/lp_20_16.txt", ref=True
    )
    gurobi_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = gurobi_solver.evaluate(calculate_gap=True)
    print(f"LPGurobiSolver Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of LP solved by LPGurobiSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)


def test_lp_gurobi_solver():
    _test_lp_gurobi_solver(True, 1)
    _test_lp_gurobi_solver(False, 2)


def test_lp():
    """
    Test LPSolver
    """
    test_lp_base_solver()
    test_lp_gurobi_solver()
    

##############################################
#             Test Func For MIS              #
##############################################

def test_mis_base_solver():
    solver = MISSolver()
    solver.from_gpickle_result_folder(
        gpickle_folder_path="tests/data_for_tests/solver/mis/mis_example/instance",
        result_folder_path="tests/data_for_tests/solver/mis/mis_example/solution",
        ref=False, cover=True
    )
    solver.to_txt("tests/data_for_tests/solver/mis/mis_example.txt")
    solver.set_solution_as_ref()
    solver.from_txt(
        file_path="tests/data_for_tests/solver/mis/mis_example.txt",
        ref=False, cover=False
    )
    gap_avg = solver.evaluate(calculate_gap=True)[2]
    if gap_avg > 1e-14:
        raise ValueError("There is a problem between txt input and read in")


def _test_mis_gurobi_solver(show_time: bool, num_threads: int):
    gurobi_solver = MISGurobiSolver(time_limit=1.0)
    gurobi_solver.from_txt(
        file_path="tests/data_for_tests/solver/mis/mis_example.txt",
        ref=True, cover=True
    )
    gurobi_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = gurobi_solver.evaluate(calculate_gap=True)
    print(f"MISGurobiSolver Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of MIS solved by MISGurobiSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)


def test_mis_gurobi_solver():
    _test_mis_gurobi_solver(True, 1)
    _test_mis_gurobi_solver(False, 2)
    
    
def test_mis_kamis_solver():
    cnf_folder_to_gpickle_folder(
        cnf_folder="tests/data_for_tests/solver/mis/mis_example_cnf/cnf",
        gpickle_foler="tests/data_for_tests/solver/mis/mis_example_cnf/mis"
    )
    kamis_solver = KaMISSolver(time_limit=10)
    kamis_solver.solve(
        src="tests/data_for_tests/solver/mis/mis_example_cnf/mis/instance", 
        out="tests/data_for_tests/solver/mis/mis_example_cnf/mis/solution"
    )
    kamis_solver.from_txt_only_sel_nodes_num(
        "tests/data_for_tests/solver/mis/mis_example_cnf/mis/ref_solution.txt", ref=True
    )
    gap_avg = kamis_solver.evaluate(calculate_gap=True)[2]
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
    test_mis_base_solver()
    test_mis_gurobi_solver()
    test_mis_kamis_solver()


##############################################
#             Test Func For MCl              #
##############################################

def _test_mcl_gurobi_solver(show_time: bool, num_threads: int):
    if not GUROBI_TEST:
        return
    gurobi_solver = MClGurobiSolver(time_limit=1.0)
    gurobi_solver.from_txt(
        file_path="tests/data_for_tests/solver/mcl/mcl_example.txt",
        ref=True, cover=True
    )
    gurobi_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = gurobi_solver.evaluate(calculate_gap=True)
    print(f"MClGurobiSolver Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of MCl solved by MClGurobiSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)


def test_mcl_gurobi_solver():
    _test_mcl_gurobi_solver(True, 1)
    _test_mcl_gurobi_solver(False, 2)


def test_mcl():
    """
    Test MClSolver
    """
    test_mcl_gurobi_solver()


##############################################
#             Test Func For MCut              #
##############################################

def _test_mcut_gurobi_solver(show_time: bool, num_threads: int):
    if not GUROBI_TEST:
        return
    gurobi_solver = MCutGurobiSolver(time_limit=1.0)
    gurobi_solver.from_txt(
        file_path="tests/data_for_tests/solver/mcut/mcut_example.txt",
        ref=True, cover=True
    )
    gurobi_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = gurobi_solver.evaluate(calculate_gap=True)
    print(f"MCutGurobiSolver Gap: {gap_avg}")
    if gap_avg >= 1e-1:
        message = (
            f"The average gap ({gap_avg}) of MCut solved by MCutGurobiSolver "
            "is larger than or equal to 1e-1%."
        )
        raise ValueError(message)


def test_mcut_gurobi_solver():
    _test_mcut_gurobi_solver(True, 1)
    _test_mcut_gurobi_solver(False, 2)


def test_mcut():
    """
    Test MCutSolver
    """
    test_mcut_gurobi_solver()
    
    
##############################################
#             Test Func For MIS              #
##############################################

def test_mvc_base_solver():
    solver = MISSolver()
    solver.from_gpickle_result_folder(
        gpickle_folder_path="tests/data_for_tests/solver/mis/mis_example/instance",
        result_folder_path="tests/data_for_tests/solver/mis/mis_example/solution",
        ref=False, cover=True
    )
    solver.to_txt("tests/data_for_tests/solver/mis/mis_example.txt")
    solver.set_solution_as_ref()
    solver.from_txt(
        file_path="tests/data_for_tests/solver/mis/mis_example.txt",
        ref=False, cover=False
    )
    gap_avg = solver.evaluate(calculate_gap=True)[2]
    if gap_avg > 1e-14:
        raise ValueError("There is a problem between txt input and read in")


def _test_mis_gurobi_solver(show_time: bool, num_threads: int):
    if not GUROBI_TEST:
        return
    gurobi_solver = MISGurobiSolver(time_limit=1.0)
    gurobi_solver.from_txt(
        file_path="tests/data_for_tests/solver/mis/mis_example.txt",
        ref=True, cover=True
    )
    gurobi_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = gurobi_solver.evaluate(calculate_gap=True)
    print(f"MISGurobiSolver Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of MIS solved by MISGurobiSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)


def test_mis_gurobi_solver():
    _test_mis_gurobi_solver(True, 1)
    _test_mis_gurobi_solver(False, 2)
    
    
def test_mis_kamis_solver():
    cnf_folder_to_gpickle_folder(
        cnf_folder="tests/data_for_tests/solver/mis/mis_example_cnf/cnf",
        gpickle_foler="tests/data_for_tests/solver/mis/mis_example_cnf/mis"
    )
    kamis_solver = KaMISSolver(time_limit=10)
    kamis_solver.solve(
        src="tests/data_for_tests/solver/mis/mis_example_cnf/mis/instance", 
        out="tests/data_for_tests/solver/mis/mis_example_cnf/mis/solution"
    )
    kamis_solver.from_txt_only_sel_nodes_num(
        "tests/data_for_tests/solver/mis/mis_example_cnf/mis/ref_solution.txt", ref=True
    )
    gap_avg = kamis_solver.evaluate(calculate_gap=True)[2]
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
    test_mis_base_solver()
    test_mis_gurobi_solver()
    test_mis_kamis_solver()


##############################################
#             Test Func For MVC              #
##############################################

def _test_mvc_gurobi_solver(show_time: bool, num_threads: int):
    if not GUROBI_TEST:
        return
    gurobi_solver = MVCGurobiSolver(time_limit=1.0)
    gurobi_solver.from_txt(
        file_path="tests/data_for_tests/solver/mvc/mvc_example.txt",
        ref=True, cover=True
    )
    gurobi_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = gurobi_solver.evaluate(calculate_gap=True)
    print(f"MVCGurobiSolver Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of MVC solved by MVCGurobiSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)


def test_mvc_gurobi_solver():
    _test_mvc_gurobi_solver(True, 1)
    _test_mvc_gurobi_solver(False, 2)


def test_mvc():
    """
    Test MVCSolver
    """
    test_mvc_gurobi_solver()


##############################################
#             Test Func For TSP              #
##############################################

def test_tsp_base_solver():
    solver = TSPSolver()
    solver.from_txt("tests/data_for_tests/solver/tsp/tsp50.txt", ref=False)
    os.remove("tests/data_for_tests/solver/tsp/tsp50.txt")
    solver.to_tsplib_folder(
        tsp_save_dir="tests/data_for_tests/solver/tsp/tsp50_tsplib_instance",
        tsp_filename="tsp50",
        tour_save_dir="tests/data_for_tests/solver/tsp/tsp50_tsplib_solution",
        tour_filename="tsp50",
        apply_scale=True,
        to_int=True,
        show_time=True
    )
    shutil.rmtree("tests/data_for_tests/solver/tsp/tsp50_tsplib_instance")
    shutil.rmtree("tests/data_for_tests/solver/tsp/tsp50_tsplib_solution")
    solver.to_txt(
        file_path="tests/data_for_tests/solver/tsp/tsp50.txt",
        apply_scale=False,
        to_int=False,
    )
    

def _test_tsp_concorde_solver(show_time: bool, num_threads: int):
    tsp_lkh_solver = TSPConcordeSolver()
    tsp_lkh_solver.from_txt("tests/data_for_tests/solver/tsp/tsp50.txt", ref=True)
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
    _test_tsp_concorde_solver(False, 2)


def _test_tsp_ga_eax_solver(show_time: bool, num_threads: int):
    tsp_ga_eax_solver = TSPGAEAXSolver()
    tsp_ga_eax_solver.from_txt("tests/data_for_tests/solver/tsp/tsp50.txt", ref=True)
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
    _test_tsp_ga_eax_solver(False, 2)


def _test_tsp_ga_eax_large_solver(show_time: bool, num_threads: int):
    tsp_ga_eax_large_solver = TSPGAEAXLargeSolver()
    tsp_ga_eax_large_solver.from_txt("tests/data_for_tests/solver/tsp/tsp1000.txt", ref=True)
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
    _test_tsp_ga_eax_large_solver(False, 2)


def _test_tsp_lkh_solver(show_time: bool, num_threads: int):
    tsp_lkh_solver = TSPLKHSolver(lkh_max_trials=100)
    tsp_lkh_solver.from_txt("tests/data_for_tests/solver/tsp/tsp50.txt", ref=True)
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
    _test_tsp_lkh_solver(False, 2)


def test_tsp():
    """
    Test TSPSolver
    """
    test_tsp_base_solver()
    test_tsp_concorde_solver()
    test_tsp_ga_eax_solver()
    test_tsp_ga_eax_large_solver()
    test_tsp_lkh_solver()

##############################################
#                    MAIN                    #
##############################################

if __name__ == "__main__":
    test_atsp()
    test_cvrp()
    test_lp()
    test_mcl()
    test_mcut()
    test_mis()
    test_mvc()
    test_tsp()
