import os
import sys
root_folder = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# sys.path.append(root_folder)
sys.path.insert(0, root_folder)
import shutil
from ml4co_kit import *

GUROBI_TEST = False
CUDA_TEST = False


##############################################
#             Test Func For ATSP             #
##############################################

def test_atsp_base_solver():
    solver = ATSPSolver()
    solver_2 = ATSPSolver()
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
    solver.to_txt(
        file_path="tests/data_for_tests/solver/atsp/atsp50.txt",
        apply_scale=False,
        to_int=False,
    )

    # test ``from_tsplib_folder``
    solver_2.from_tsplib_folder(
        atsp_folder_path="tests/data_for_tests/solver/atsp/atsp50_tsplib_instance",
    )
    solver_2.from_tsplib_folder(
        tour_folder_path="tests/data_for_tests/solver/atsp/atsp50_tsplib_solution",
    )
    solver_2.from_tsplib_folder(
        atsp_folder_path="tests/data_for_tests/solver/atsp/atsp50_tsplib_instance",
        tour_folder_path="tests/data_for_tests/solver/atsp/atsp50_tsplib_solution",
    )
    shutil.rmtree("tests/data_for_tests/solver/atsp/atsp50_tsplib_instance")
    shutil.rmtree("tests/data_for_tests/solver/atsp/atsp50_tsplib_solution")

    
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
    

def _test_atsp_or_solver(show_time: bool, num_threads: int):
    atsp_or_solver = ATSPORSolver(time_limit=1)
    atsp_or_solver.from_txt("tests/data_for_tests/solver/atsp/atsp50.txt", ref=True)
    atsp_or_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = atsp_or_solver.evaluate(calculate_gap=True)
    print(f"ATSPORSolver Gap: {gap_avg}")
    if gap_avg >= 30:
        message = (
            f"The average gap ({gap_avg}) of ATSP50 solved by ATSPORSolver "
            "is larger than or equal to 30%."
        )
        raise ValueError(message)


def test_atsp_or_solver():
    _test_atsp_or_solver(True, 1)
    _test_atsp_or_solver(False, 2)
    
       
def test_atsp():
    """
    Test ATSPSolver
    """
    test_atsp_base_solver()
    # test_atsp_lkh_solver()
    # test_atsp_or_solver()


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
    if gap_avg >= 1:
        message = (
            f"The average gap ({gap_avg}) of CVRP50 solved by CVRPPyVRPSolver "
            "is larger than or equal to 1%."
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
#              Test Func For KP              #
##############################################


def test_kp_base_solver():
    solver = KPSolver()
    solver.from_txt("tests/data_for_tests/solver/kp/kp100_example.txt", ref=False)
    os.remove("tests/data_for_tests/solver/kp/kp100_example.txt")
    solver.to_txt("tests/data_for_tests/solver/kp/kp100_example.txt")
    

def _test_kp_or_solver(show_time: bool, num_threads: int):
    or_solver = KPORSolver(time_limit=10.0)
    or_solver.from_txt(
        file_path="tests/data_for_tests/solver/kp/kp100_example.txt", ref=True
    )
    or_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = or_solver.evaluate(calculate_gap=True)
    print(f"KPORSolver Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of KP solved by KPORSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)


def test_kp_or_solver():
    _test_kp_or_solver(True, 1)
    _test_kp_or_solver(False, 2)


def test_kp():
    """
    Test KPSolver
    """
    test_kp_base_solver()
    test_kp_or_solver()
    

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
    costs = solver.evaluate(calculate_gap=False)
    print(f"avrage costs of mis examples: {costs}")
    if gap_avg > 1e-14:
        raise ValueError("There is a problem between txt input and read in")
    solver.to_gpickle_result_folder(
        gpickle_save_dir="tmp/instance", gpickle_filename="mis_example",
        result_save_dir="tmp/solution", result_filename="mis_example"
    )

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


def _test_mcl_or_solver(show_time: bool, num_threads: int):
    or_solver = MClORSolver(time_limit=5.0)
    or_solver.from_txt(
        file_path="tests/data_for_tests/solver/mcl/mcl_example.txt",
        ref=True, cover=True
    )
    or_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = or_solver.evaluate(calculate_gap=True)
    print(f"MClORSolver Gap: {gap_avg}")
    if gap_avg >= 1:
        message = (
            f"The average gap ({gap_avg}) of MCl solved by MClORSolver "
            "is larger than or equal to 1%."
        )
        raise ValueError(message)


def test_mcl_or_solver():
    _test_mcl_or_solver(True, 1)
    _test_mcl_or_solver(False, 2)
    
    
def test_mcl():
    """
    Test MClSolver
    """
    test_mcl_gurobi_solver()
    test_mcl_or_solver()


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


def _test_mcut_or_solver(show_time: bool, num_threads: int):
    or_solver = MCutORSolver(time_limit=1.0)
    or_solver.from_txt(
        file_path="tests/data_for_tests/solver/mcut/mcut_example.txt",
        ref=True, cover=True
    )
    or_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = or_solver.evaluate(calculate_gap=True)
    print(f"MCutORSolver Gap: {gap_avg}")
    if gap_avg >= 30:
        message = (
            f"The average gap ({gap_avg}) of MCut solved by MCutORSolver "
            "is larger than or equal to 30%."
        )
        raise ValueError(message)


def test_mcut_or_solver():
    _test_mcut_or_solver(True, 1)
    _test_mcut_or_solver(False, 2)
    
    
def test_mcut():
    """
    Test MCutSolver
    """
    test_mcut_gurobi_solver()
    test_mcut_or_solver()
    
    
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


def _test_mis_or_solver(show_time: bool, num_threads: int):
    or_solver = MISORSolver(time_limit=5.0)
    or_solver.from_txt(
        file_path="tests/data_for_tests/solver/mis/mis_example.txt",
        ref=True, cover=True
    )
    or_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = or_solver.evaluate(calculate_gap=True)
    print(f"MISORSolver Gap: {gap_avg}")
    if gap_avg >= 1:
        message = (
            f"The average gap ({gap_avg}) of MIS solved by MISORSolver "
            "is larger than or equal to 1%."
        )
        raise ValueError(message)


def test_mis_or_solver():
    _test_mis_or_solver(True, 1)
    _test_mis_or_solver(False, 2)


def test_mis():
    """
    Test MISSolver
    """
    test_mis_base_solver()
    test_mis_gurobi_solver()
    test_mis_kamis_solver()
    test_mis_or_solver()


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


def _test_mvc_or_solver(show_time: bool, num_threads: int):
    or_solver = MVCORSolver(time_limit=1.0)
    or_solver.from_txt(
        file_path="tests/data_for_tests/solver/mvc/mvc_example.txt",
        ref=True, cover=True
    )
    or_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = or_solver.evaluate(calculate_gap=True)
    print(f"MVCORSolver Gap: {gap_avg}")
    if gap_avg >= 1:
        message = (
            f"The average gap ({gap_avg}) of MVC solved by MVCORSolver "
            "is larger than or equal to 1%."
        )
        raise ValueError(message)


def test_mvc_or_solver():
    _test_mvc_or_solver(True, 1)
    _test_mvc_or_solver(False, 2)
    

def test_mvc():
    """
    Test MVCSolver
    """
    test_mvc_gurobi_solver()
    test_mvc_or_solver()
    
    
##############################################
#              Test Func For OP              #
##############################################


def test_op_base_solver():
    solver = OPSolver()
    solver.from_txt("tests/data_for_tests/solver/op/op_example.txt", ref=False)
    os.remove("tests/data_for_tests/solver/op/op_example.txt")
    solver.to_txt("tests/data_for_tests/solver/op/op_example.txt")
    

def _test_op_gurobi_solver(show_time: bool, num_threads: int):
    or_solver = OPGurobiSolver(time_limit=10.0)
    or_solver.from_txt(
        file_path="tests/data_for_tests/solver/op/op_example.txt", ref=True
    )
    or_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = or_solver.evaluate(calculate_gap=True)
    print(f"OPGurobiSolver Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of OP solved by OPGurobiSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)


def test_op_gurobi_solver():
    _test_op_gurobi_solver(True, 1)
    _test_op_gurobi_solver(False, 2)


def test_op():
    """
    Test OPSolver
    """
    test_op_base_solver()
    test_op_gurobi_solver()


##############################################
#             Test Func For PCTSP            #
##############################################

def test_pctsp_base_solver():
    solver = PCTSPSolver()
    solver.from_txt("tests/data_for_tests/solver/pctsp/pctsp_example.txt", ref=False)
    os.remove("tests/data_for_tests/solver/pctsp/pctsp_example.txt")
    solver.to_txt(file_path="tests/data_for_tests/solver/pctsp/pctsp_example.txt")
    

def _test_pctsp_ils_solver(show_time: bool, num_threads: int):
    pctsp_ils_solver = PCTSPILSSolver()
    pctsp_ils_solver.from_txt("tests/data_for_tests/solver/pctsp/pctsp_example.txt", ref=True)
    pctsp_ils_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = pctsp_ils_solver.evaluate(calculate_gap=True)
    print(f"PCTSPILSSolver Gap: {gap_avg}")
    if gap_avg >= 1e-3:
        message = (
            f"The average gap ({gap_avg}) of PCTSP50 solved by PCTSPILSSolver "
            "is larger than or equal to 1e-3%."
        )
        raise ValueError(message)


def test_pctsp_ils_solver():
    _test_pctsp_ils_solver(True, 1)
    _test_pctsp_ils_solver(False, 2)


def _test_pctsp_or_solver(show_time: bool, num_threads: int):
    pctsp_or_solver = PCTSPORSolver()
    pctsp_or_solver.from_txt("tests/data_for_tests/solver/pctsp/pctsp_example.txt", ref=True)
    pctsp_or_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = pctsp_or_solver.evaluate(calculate_gap=True)
    print(f"TSPORSolver Gap: {gap_avg}")
    if gap_avg >= 5:
        message = (
            f"The average gap ({gap_avg}) of PCTSP50 solved by TSPORSolver "
            "is larger than or equal to 5%."
        )
        raise ValueError(message)
    
    
def test_pctsp_or_solver():
    _test_pctsp_or_solver(True, 1)
    _test_pctsp_or_solver(False, 2)
    
    
def test_pctsp():
    """
    Test PCTSPSolver
    """
    test_pctsp_base_solver()
    test_pctsp_ils_solver()
    test_pctsp_or_solver()
    
    
##############################################
#             Test Func For SPCTSP           #
##############################################

def test_spctsp_base_solver():
    solver = SPCTSPSolver()
    solver.from_txt("tests/data_for_tests/solver/spctsp/spctsp_example.txt", ref=False)
    os.remove("tests/data_for_tests/solver/spctsp/spctsp_example.txt")
    solver.to_txt(file_path="tests/data_for_tests/solver/spctsp/spctsp_example.txt")
    

def _test_spctsp_reopt_solver(show_time: bool, num_threads: int):
    spctsp_reopt_solver = SPCTSPReoptSolver()
    spctsp_reopt_solver.from_txt("tests/data_for_tests/solver/spctsp/spctsp_example.txt", ref=True)
    spctsp_reopt_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = spctsp_reopt_solver.evaluate(calculate_gap=True)
    print(f"SPCTSPReoptSolver Gap: {gap_avg}")
    if gap_avg >= 1e-3:
        message = (
            f"The average gap ({gap_avg}) of SPCTSP50 solved by SPCTSPReoptSolver "
            "is larger than or equal to 1e-3%."
        )
        raise ValueError(message)


def test_spctsp_reopt_solver():
    _test_spctsp_reopt_solver(True, 1)
    _test_spctsp_reopt_solver(False, 2)
    
    
def test_spctsp():
    """
    Test SPCTSPSolver
    """
    test_spctsp_base_solver()
    test_spctsp_reopt_solver()


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


def _test_tsp_neurolkh_solver(show_time: bool, num_threads: int, batch_size: int):
    # use nn
    if CUDA_TEST:
        tsp_lkh_solver = TSPNeuroLKHSolver(
            lkh_max_trials=100,
            use_nn=True, 
            sparse_factor=20,
            neurolkh_device="cuda" 
        )
        tsp_lkh_solver.from_txt("tests/data_for_tests/solver/tsp/tsp50.txt", ref=True)
        tsp_lkh_solver.solve(show_time=show_time, num_threads=num_threads, batch_size=batch_size)
        _, _, gap_avg, _ = tsp_lkh_solver.evaluate(calculate_gap=True)
        print(f"TSPNeuroLKHSolver(with nn) Gap: {gap_avg}")
        if gap_avg >= 1e-2:
            message = (
                f"The average gap ({gap_avg}) of TSP50 solved by TSPNeuroLKHSolver "
                "is larger than or equal to 1e-2%."
            )
            raise ValueError(message)
    
    # not use nn
    tsp_lkh_solver = TSPNeuroLKHSolver(
        lkh_max_trials=100, 
        use_nn=False, 
        sparse_factor=20, 
    )
    tsp_lkh_solver.from_txt("tests/data_for_tests/solver/tsp/tsp50.txt", ref=True)
    tsp_lkh_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = tsp_lkh_solver.evaluate(calculate_gap=True)
    print(f"TSPNeuroLKHSolver(without nn) Gap: {gap_avg}")
    if gap_avg >= 1e-2:
        message = (
            f"The average gap ({gap_avg}) of TSP50 solved by TSPNeuroLKHSolver "
            "is larger than or equal to 1e-2%."
        )
        raise ValueError(message)


def test_tsp_neurolkh_solver():
    _test_tsp_neurolkh_solver(True, 1, 4)
    _test_tsp_neurolkh_solver(False, 2, 4)


def _test_tsp_or_solver(show_time: bool, num_threads: int):
    tsp_or_solver = TSPORSolver()
    tsp_or_solver.from_txt("tests/data_for_tests/solver/tsp/tsp50.txt", ref=True)
    tsp_or_solver.solve(show_time=show_time, num_threads=num_threads)
    _, _, gap_avg, _ = tsp_or_solver.evaluate(calculate_gap=True)
    print(f"TSPORSolver Gap: {gap_avg}")
    if gap_avg >= 5:
        message = (
            f"The average gap ({gap_avg}) of TSP50 solved by TSPORSolver "
            "is larger than or equal to 5%."
        )
        raise ValueError(message)
    
    
def test_tsp_or_solver():
    _test_tsp_or_solver(True, 1)
    _test_tsp_or_solver(False, 2)
    
    
def test_tsp():
    """
    Test TSPSolver
    """
    test_tsp_base_solver()
    test_tsp_concorde_solver()
    test_tsp_ga_eax_solver()
    test_tsp_ga_eax_large_solver()
    test_tsp_lkh_solver()
    test_tsp_neurolkh_solver()
    test_tsp_or_solver()


##############################################
#                    MAIN                    #
##############################################

if __name__ == "__main__":
    test_atsp()
    # test_cvrp()
    # test_kp()
    # test_lp()
    # test_mcl()
    # test_mcut()
    # test_mis()
    # test_mvc()
    # test_op()
    # test_pctsp()
    # test_spctsp()
    # test_tsp()